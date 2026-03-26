"""Main SENTINEL Inference Monitor -- async event loop orchestrator.

Runs as a sidecar process that:
1. Intercepts sampled inference output tensors via the configured interceptor.
2. Analyzes tensors through a pipeline of statistical analyzers.
3. Reports detected anomalies to the Correlation Engine via gRPC.
4. Exposes a health-check HTTP server for Kubernetes liveness/readiness.

Concurrency model:
    Main async event loop (uvloop on Linux)
    +-- Interceptor coroutine: captures tensors -> asyncio.Queue
    +-- N Analyzer workers: consume from queue, run analysis pipeline
    +-- gRPC reporter: batches and sends anomaly events
    +-- Health-check HTTP server: /healthz and /readyz endpoints
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
import time
from typing import Any

import numpy as np
import structlog
from aiohttp import web

from sentinel_inference.analyzers.entropy_analyzer import EntropyAnalyzer
from sentinel_inference.analyzers.kl_divergence import KLDivergenceDetector
from sentinel_inference.analyzers.logit_analyzer import AnomalyEvent, LogitAnalyzer
from sentinel_inference.analyzers.spectral_analyzer import SpectralAnalyzer
from sentinel_inference.analyzers.statistical_tests import StatisticalTestAnalyzer
from sentinel_inference.config import MonitorConfig
from sentinel_inference.grpc_client import CorrelationEngineClient
from sentinel_inference.interceptors.base import BaseInterceptor, TensorCapture
from sentinel_inference.interceptors.generic_interceptor import GenericShmInterceptor
from sentinel_inference.interceptors.triton_interceptor import TritonInterceptor
from sentinel_inference.interceptors.trtllm_interceptor import TrtLLMInterceptor
from sentinel_inference.interceptors.vllm_interceptor import VLLMInterceptor
from sentinel_inference.metrics import (
    ANALYSIS_DURATION,
    ANOMALIES_TOTAL,
    DROPPED_SAMPLES_TOTAL,
    QUEUE_SIZE,
    SAMPLES_TOTAL,
    start_metrics_server,
)
from sentinel_inference.signatures.sketch import CountMinSketch, HyperLogLogSketch
from sentinel_inference.signatures.tensor_fingerprint import TensorFingerprint

logger = structlog.get_logger(__name__)


def _build_interceptor(config: MonitorConfig) -> BaseInterceptor:
    """Factory: build the appropriate interceptor from config."""
    itype = config.interceptor.type
    common = dict(gpu_id=config.gpu_id, sample_rate=config.sampling.rate)
    if itype == "triton":
        return TritonInterceptor(
            shm_name=config.interceptor.shm_name,
            shm_size=config.interceptor.shm_size_bytes,
            **common,
        )
    elif itype == "vllm":
        return VLLMInterceptor(**common)
    elif itype == "trtllm":
        return TrtLLMInterceptor(**common)
    else:
        return GenericShmInterceptor(
            shm_name=config.interceptor.shm_name,
            shm_size=config.interceptor.shm_size_bytes,
            **common,
        )


class InferenceMonitor:
    """Main orchestrator for the SENTINEL inference monitoring sidecar.

    Parameters
    ----------
    config : MonitorConfig
        Full configuration for all sub-components.
    """

    def __init__(self, config: MonitorConfig) -> None:
        self._config = config
        self._queue: asyncio.Queue[TensorCapture] = asyncio.Queue(
            maxsize=config.sampling.max_queue_size
        )
        self._interceptor = _build_interceptor(config)
        self._grpc_client = CorrelationEngineClient(
            config=config.grpc,
            node_id=config.node_id,
            gpu_id=config.gpu_id,
            replica_id=config.replica_id,
        )

        # Analyzers
        self._logit_analyzer = LogitAnalyzer(config.logit_analyzer)
        self._entropy_analyzer = EntropyAnalyzer(config.entropy_analyzer)
        self._kl_detector = KLDivergenceDetector(
            config=config.kl_divergence,
            replica_id=config.replica_id,
        )
        self._spectral_analyzer = SpectralAnalyzer(config.spectral_analyzer)
        self._stat_tests = StatisticalTestAnalyzer(config.statistical_tests)

        # Signatures
        self._fingerprinter = TensorFingerprint(
            projection_dim=config.fingerprint.projection_dim,
            bits_per_dim=config.fingerprint.bits_per_dim,
            seed=config.fingerprint.seed,
        )
        self._cms = CountMinSketch(
            width=config.sketch.cms_width,
            depth=config.sketch.cms_depth,
        )
        self._hll = HyperLogLogSketch(precision=config.sketch.hll_precision)

        self._running = False
        self._tasks: list[asyncio.Task[Any]] = []
        self._health_app: web.Application | None = None
        self._health_runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start all sub-components and worker tasks."""
        logger.info(
            "monitor_starting",
            node_id=self._config.node_id,
            gpu_id=self._config.gpu_id,
            interceptor=self._config.interceptor.type,
            sample_rate=self._config.sampling.rate,
            num_workers=self._config.sampling.num_workers,
        )

        self._running = True

        # Start metrics server
        start_metrics_server(
            host=self._config.metrics.host,
            port=self._config.metrics.port,
        )

        # Start interceptor
        await self._interceptor.start()

        # Start gRPC client
        await self._grpc_client.start()

        # Start health check server
        await self._start_health_server()

        # Launch interceptor producer task
        self._tasks.append(asyncio.create_task(self._intercept_loop()))

        # Launch analyzer worker tasks
        for i in range(self._config.sampling.num_workers):
            self._tasks.append(asyncio.create_task(self._analyzer_worker(i)))

        logger.info("monitor_started")

    async def stop(self) -> None:
        """Gracefully shut down all tasks and sub-components."""
        logger.info("monitor_stopping")
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        await self._interceptor.stop()
        await self._grpc_client.stop()

        if self._health_runner is not None:
            await self._health_runner.cleanup()

        logger.info("monitor_stopped")

    async def run_forever(self) -> None:
        """Start and run until interrupted."""
        await self.start()
        try:
            # Wait until _running is set to False
            while self._running:
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # ------------------------------------------------------------------
    # Interceptor producer
    # ------------------------------------------------------------------

    async def _intercept_loop(self) -> None:
        """Continuously capture tensors and enqueue them for analysis."""
        labels = {
            "node_id": self._config.node_id,
            "gpu_id": str(self._config.gpu_id),
        }
        while self._running:
            try:
                capture = await self._interceptor.capture_output()
                if capture is None:
                    continue

                SAMPLES_TOTAL.labels(**labels).inc()

                try:
                    self._queue.put_nowait(capture)
                    QUEUE_SIZE.labels(**labels).set(self._queue.qsize())
                except asyncio.QueueFull:
                    DROPPED_SAMPLES_TOTAL.labels(**labels).inc()
                    logger.warning("sample_dropped_queue_full")

            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("intercept_loop_error")
                await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Analyzer workers
    # ------------------------------------------------------------------

    async def _analyzer_worker(self, worker_id: int) -> None:
        """Consume captures from the queue and run the analysis pipeline."""
        labels = {
            "node_id": self._config.node_id,
            "gpu_id": str(self._config.gpu_id),
        }
        while self._running:
            try:
                capture = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            QUEUE_SIZE.labels(**labels).set(self._queue.qsize())

            all_anomalies: list[AnomalyEvent] = []

            try:
                all_anomalies.extend(await self._run_analysis(capture, labels))
            except Exception:
                logger.exception("analyzer_worker_error", worker_id=worker_id)

            if all_anomalies:
                for event in all_anomalies:
                    ANOMALIES_TOTAL.labels(
                        **labels,
                        type=event.analyzer,
                    ).inc()
                await self._grpc_client.submit(all_anomalies)

    async def _run_analysis(
        self,
        capture: TensorCapture,
        labels: dict[str, str],
    ) -> list[AnomalyEvent]:
        """Run all analyzers on a captured tensor.

        Each analyzer is run in the default executor to avoid blocking
        the event loop (they are CPU-bound NumPy operations).
        """
        loop = asyncio.get_running_loop()
        anomalies: list[AnomalyEvent] = []

        # Logit analyzer
        t0 = time.monotonic()
        result = await loop.run_in_executor(None, self._logit_analyzer.analyze, capture.tensor)
        ANALYSIS_DURATION.labels(**labels, analyzer="logit").observe(time.monotonic() - t0)
        anomalies.extend(result)

        # Entropy analyzer
        t0 = time.monotonic()
        result = await loop.run_in_executor(None, self._entropy_analyzer.analyze, capture.tensor)
        ANALYSIS_DURATION.labels(**labels, analyzer="entropy").observe(time.monotonic() - t0)
        anomalies.extend(result)

        # KL divergence (cross-replica)
        t0 = time.monotonic()
        result = await loop.run_in_executor(
            None,
            self._kl_detector.submit,
            capture.tensor,
            capture.input_hash,
            capture.request_id,
            None,
        )
        ANALYSIS_DURATION.labels(**labels, analyzer="kl_divergence").observe(time.monotonic() - t0)
        anomalies.extend(result)

        # Spectral analyzer
        t0 = time.monotonic()
        result = await loop.run_in_executor(None, self._spectral_analyzer.analyze, capture.tensor)
        ANALYSIS_DURATION.labels(**labels, analyzer="spectral").observe(time.monotonic() - t0)
        anomalies.extend(result)

        # Statistical tests
        t0 = time.monotonic()
        result = await loop.run_in_executor(None, self._stat_tests.analyze, capture.tensor)
        ANALYSIS_DURATION.labels(**labels, analyzer="statistical_tests").observe(
            time.monotonic() - t0
        )
        anomalies.extend(result)

        # Tensor fingerprint (for logging / future use)
        _fp = await loop.run_in_executor(None, self._fingerprinter.compute, capture.tensor)

        # Update token sketches if tensor looks like token IDs
        if capture.tensor.ndim >= 1:
            top_indices = np.argsort(capture.tensor.ravel())[-10:]
            for idx in top_indices:
                self._cms.add(int(idx))
                self._hll.add(int(idx))

        return anomalies

    # ------------------------------------------------------------------
    # Health check server
    # ------------------------------------------------------------------

    async def _start_health_server(self) -> None:
        """Start a lightweight HTTP server for K8s probes."""
        app = web.Application()
        app.router.add_get("/healthz", self._handle_healthz)
        app.router.add_get("/readyz", self._handle_readyz)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            self._config.health_check.host,
            self._config.health_check.port,
        )
        await site.start()
        self._health_app = app
        self._health_runner = runner
        logger.info(
            "health_server_started",
            port=self._config.health_check.port,
        )

    async def _handle_healthz(self, request: web.Request) -> web.Response:
        """Liveness probe -- always OK if the process is running."""
        return web.json_response({"status": "ok"})

    async def _handle_readyz(self, request: web.Request) -> web.Response:
        """Readiness probe -- OK once the interceptor is connected."""
        if self._running:
            return web.json_response({"status": "ready"})
        return web.json_response({"status": "not_ready"}, status=503)


def main() -> None:
    """CLI entry point for the SENTINEL Inference Monitor."""
    import argparse

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    parser = argparse.ArgumentParser(description="SENTINEL Inference Monitor")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config = MonitorConfig.from_yaml(args.config) if args.config else MonitorConfig.from_env()

    # Use uvloop on Linux for better performance
    if sys.platform != "win32":
        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("using_uvloop")
        except ImportError:
            pass

    monitor = InferenceMonitor(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(monitor.stop()))

    try:
        loop.run_until_complete(monitor.run_forever())
    except KeyboardInterrupt:
        loop.run_until_complete(monitor.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
