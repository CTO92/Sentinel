"""
SENTINEL Benchmark — Correlation Engine Load Test

Simulates 100-10,000 agents concurrently sending events to the correlation
engine to measure ingestion throughput, correlation latency, and memory usage.

Supports two modes:
  - **gRPC mode**: Uses the generated sentinel.v1 protobuf stubs to send
    real ProbeResultBatch and AnomalyBatch messages via the bidirectional
    streaming RPCs (ProbeService/StreamProbeResults, AnomalyService/StreamAnomalyEvents).
  - **Simulation mode**: When gRPC is unavailable or the endpoint is unreachable,
    simulates realistic network + processing latency for capacity planning.

Usage:
    python correlation_engine_load.py --endpoint localhost:50051
    python correlation_engine_load.py --endpoint localhost:50051 --agents 5000 --duration 300
    python correlation_engine_load.py --endpoint localhost:50051 --output load_test.json

Copyright 2025-2026 SENTINEL Authors — Apache 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import struct
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class LoadTestResult:
    """Result of a load test at a specific agent count."""
    agent_count: int
    duration_seconds: float
    total_events_sent: int
    total_events_acked: int
    events_per_second: float
    send_latency_p50_ms: float
    send_latency_p95_ms: float
    send_latency_p99_ms: float
    send_latency_mean_ms: float
    correlation_latency_p50_ms: float
    correlation_latency_p95_ms: float
    correlation_latency_p99_ms: float
    errors: int
    error_rate_pct: float
    memory_usage_mb: float
    mode: str  # "grpc" or "simulated"


class AgentSimulator:
    """Simulates a single probe agent sending events."""

    PROBE_TYPES = ["fma", "tensor_core", "transcendental", "aes", "memory",
                   "register_file", "shared_memory"]

    def __init__(
        self,
        agent_id: str,
        gpu_uuid: str,
        node_id: str,
        event_rate_hz: float = 1.0,
        sdc_probability: float = 0.001,
    ):
        self.agent_id = agent_id
        self.gpu_uuid = gpu_uuid
        self.node_id = node_id
        self.event_rate_hz = event_rate_hz
        self.sdc_probability = sdc_probability
        self._rng = random.Random(hash(agent_id))
        self.send_latencies: list[float] = []
        self.events_sent = 0
        self.events_acked = 0
        self.errors = 0
        self._stop = False
        self._sequence_number = 0

    def generate_probe_event(self) -> dict[str, Any]:
        """Generate a synthetic probe result matching ProbeExecution proto."""
        has_sdc = self._rng.random() < self.sdc_probability
        probe_type = self._rng.choice(self.PROBE_TYPES)
        sm_id = self._rng.randint(0, 131)

        # Generate deterministic hash bytes.
        expected_hash = bytes(range(32))
        actual_hash = expected_hash if not has_sdc else bytes(
            (b + 1) % 256 for b in expected_hash
        )

        return {
            "execution_id": str(uuid.uuid4()),
            "probe_type": probe_type,
            "sm": {
                "gpu": {
                    "uuid": self.gpu_uuid,
                    "hostname": self.node_id,
                    "device_index": 0,
                },
                "sm_id": sm_id,
            },
            "result": "fail" if has_sdc else "pass",
            "expected_hash": expected_hash.hex(),
            "actual_hash": actual_hash.hex(),
            "execution_time_ns": self._rng.randint(50_000, 500_000),
            "gpu_clock_mhz": self._rng.randint(1400, 2100),
            "gpu_temperature_c": self._rng.uniform(40.0, 85.0),
            "gpu_power_w": self._rng.uniform(200.0, 700.0),
            "timestamp": time.time(),
        }

    def generate_anomaly_event(self) -> dict[str, Any]:
        """Generate a synthetic anomaly event matching AnomalyEvent proto."""
        anomaly_types = [
            ("logit_drift", 1), ("entropy_anomaly", 2), ("kl_divergence", 3),
            ("gradient_spike", 4), ("loss_spike", 5),
        ]
        name, type_val = self._rng.choice(anomaly_types)
        score = self._rng.uniform(3.0, 10.0)
        threshold = self._rng.uniform(2.0, 4.0)

        return {
            "event_id": str(uuid.uuid4()),
            "anomaly_type": type_val,
            "source": 1,  # INFERENCE_MONITOR
            "gpu": {
                "uuid": self.gpu_uuid,
                "hostname": self.node_id,
                "device_index": 0,
            },
            "severity": self._rng.choice([1, 2, 3, 4]),
            "score": score,
            "threshold": threshold,
            "details": f"Simulated {name}: score={score:.3f} > threshold={threshold:.3f}",
            "timestamp": time.time(),
            "step_number": self._rng.randint(0, 100_000),
        }

    def run(self, duration_s: float, send_fn: Callable[[dict[str, Any]], None]) -> None:
        """Run the agent for a given duration."""
        interval = 1.0 / self.event_rate_hz
        end_time = time.time() + duration_s

        while time.time() < end_time and not self._stop:
            # Alternate between probe results and anomaly events.
            if self._rng.random() < 0.7:
                event = self.generate_probe_event()
            else:
                event = self.generate_anomaly_event()

            start = time.time()
            try:
                send_fn(event)
                latency_ms = (time.time() - start) * 1000
                self.send_latencies.append(latency_ms)
                self.events_sent += 1
                self.events_acked += 1
            except Exception:
                self.errors += 1
                self.events_sent += 1

            # Rate limiting.
            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        self._stop = True


class GrpcSender:
    """Manages gRPC connections and sends events using protobuf stubs."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._channel: Any = None
        self._probe_stub: Any = None
        self._anomaly_stub: Any = None
        self._lock = threading.Lock()
        self._probe_batch_buffer: list[dict[str, Any]] = []
        self._anomaly_batch_buffer: list[dict[str, Any]] = []
        self._sequence_number = 0
        self._proto_available = False
        self._grpc_available = False

        self._connect()

    def _connect(self) -> None:
        """Establish the gRPC channel and create service stubs."""
        try:
            import grpc
            self._channel = grpc.insecure_channel(
                self.endpoint,
                options=[
                    ("grpc.keepalive_time_ms", 10_000),
                    ("grpc.keepalive_timeout_ms", 5_000),
                    ("grpc.max_send_message_length", 16 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                ],
            )
            self._grpc_available = True
        except ImportError:
            print("  [warn] grpcio not installed. Running in simulation mode.")
            return

        # Try to import generated stubs.
        try:
            from sentinel.v1 import anomaly_pb2_grpc  # type: ignore[import-untyped]
            from sentinel.v1 import probe_pb2_grpc  # type: ignore[import-untyped]
            self._probe_stub = probe_pb2_grpc.ProbeServiceStub(self._channel)
            self._anomaly_stub = anomaly_pb2_grpc.AnomalyServiceStub(self._channel)
            self._proto_available = True
        except ImportError:
            # Fall back to generic gRPC calls.
            self._proto_available = False

    def send_event(self, event: dict[str, Any]) -> None:
        """Send a single event via gRPC.

        For probe events, uses ProbeService/StreamProbeResults.
        For anomaly events, uses AnomalyService/StreamAnomalyEvents.
        Falls back to generic unary calls when proto stubs are unavailable.
        """
        if not self._grpc_available:
            raise RuntimeError("gRPC not available")

        with self._lock:
            self._sequence_number += 1
            seq = self._sequence_number

        is_anomaly = "anomaly_type" in event

        if self._proto_available:
            self._send_with_proto_stubs(event, seq, is_anomaly)
        else:
            self._send_with_generic_call(event, seq, is_anomaly)

    def _send_with_proto_stubs(self, event: dict[str, Any], seq: int, is_anomaly: bool) -> None:
        """Send using generated protobuf stubs."""
        import grpc  # type: ignore[import-untyped]

        if is_anomaly:
            from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore[import-untyped]
            from sentinel.v1 import anomaly_pb2  # type: ignore[import-untyped]
            from sentinel.v1 import common_pb2  # type: ignore[import-untyped]

            batch = anomaly_pb2.AnomalyBatch()
            batch.source_hostname = event.get("gpu", {}).get("hostname", "loadtest")
            batch.sequence_number = seq
            ts = Timestamp()
            ts.FromSeconds(int(time.time()))
            batch.batch_timestamp.CopyFrom(ts)

            evt = anomaly_pb2.AnomalyEvent()
            evt.event_id = event["event_id"]
            evt.anomaly_type = event["anomaly_type"]
            evt.source = event.get("source", 1)

            gpu = common_pb2.GpuIdentifier()
            gpu_data = event.get("gpu", {})
            gpu.uuid = gpu_data.get("uuid", "")
            gpu.hostname = gpu_data.get("hostname", "")
            gpu.device_index = gpu_data.get("device_index", 0)
            evt.gpu.CopyFrom(gpu)

            evt.severity = event.get("severity", 2)
            evt.score = event.get("score", 0.0)
            evt.threshold = event.get("threshold", 0.0)
            evt.details = event.get("details", "")
            evt.step_number = event.get("step_number", 0)

            evt_ts = Timestamp()
            evt_ts.FromSeconds(int(event.get("timestamp", time.time())))
            evt.timestamp.CopyFrom(evt_ts)

            batch.events.append(evt)

            # Use unary call wrapping the batch for load testing.
            call = self._channel.unary_unary(
                "/sentinel.v1.AnomalyService/StreamAnomalyEvents",
                request_serializer=batch.SerializeToString,
                response_deserializer=lambda x: x,
            )
            call(batch, timeout=5.0)
        else:
            from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore[import-untyped]
            from sentinel.v1 import common_pb2  # type: ignore[import-untyped]
            from sentinel.v1 import probe_pb2  # type: ignore[import-untyped]

            batch = probe_pb2.ProbeResultBatch()
            batch.agent_hostname = event.get("sm", {}).get("gpu", {}).get("hostname", "loadtest")
            batch.sequence_number = seq
            ts = Timestamp()
            ts.FromSeconds(int(time.time()))
            batch.batch_timestamp.CopyFrom(ts)

            exe = probe_pb2.ProbeExecution()
            exe.execution_id = event["execution_id"]

            probe_type_map = {
                "fma": 1, "tensor_core": 2, "transcendental": 3, "aes": 4,
                "memory": 5, "register_file": 6, "shared_memory": 7,
            }
            exe.probe_type = probe_type_map.get(event.get("probe_type", ""), 0)

            sm_data = event.get("sm", {})
            sm = common_pb2.SmIdentifier()
            gpu = common_pb2.GpuIdentifier()
            gpu_data = sm_data.get("gpu", {})
            gpu.uuid = gpu_data.get("uuid", "")
            gpu.hostname = gpu_data.get("hostname", "")
            gpu.device_index = gpu_data.get("device_index", 0)
            sm.gpu.CopyFrom(gpu)
            sm.sm_id = sm_data.get("sm_id", 0)
            exe.sm.CopyFrom(sm)

            result_map = {"pass": 1, "fail": 2, "error": 3, "timeout": 4}
            exe.result = result_map.get(event.get("result", ""), 0)

            exe.expected_hash = bytes.fromhex(event.get("expected_hash", "00" * 32))
            exe.actual_hash = bytes.fromhex(event.get("actual_hash", "00" * 32))
            exe.execution_time_ns = event.get("execution_time_ns", 0)
            exe.gpu_clock_mhz = event.get("gpu_clock_mhz", 0)
            exe.gpu_temperature_c = event.get("gpu_temperature_c", 0.0)
            exe.gpu_power_w = event.get("gpu_power_w", 0.0)

            exe_ts = Timestamp()
            exe_ts.FromSeconds(int(event.get("timestamp", time.time())))
            exe.timestamp.CopyFrom(exe_ts)

            batch.executions.append(exe)

            call = self._channel.unary_unary(
                "/sentinel.v1.ProbeService/StreamProbeResults",
                request_serializer=batch.SerializeToString,
                response_deserializer=lambda x: x,
            )
            call(batch, timeout=5.0)

    def _send_with_generic_call(self, event: dict[str, Any], seq: int, is_anomaly: bool) -> None:
        """Send using generic gRPC call with JSON serialization."""
        payload = json.dumps({
            "sequence_number": seq,
            "timestamp": time.time(),
            "event": event,
        }, default=str).encode("utf-8")

        method = (
            "/sentinel.v1.AnomalyService/StreamAnomalyEvents"
            if is_anomaly
            else "/sentinel.v1.ProbeService/StreamProbeResults"
        )

        call = self._channel.unary_unary(
            method,
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )
        call(payload, timeout=5.0)

    def close(self) -> None:
        if self._channel:
            self._channel.close()


class CorrelationEngineLoadTest:
    """
    Load test framework for the SENTINEL Correlation Engine.

    Operates in two modes:
    - **gRPC mode**: Sends real protobuf messages to a running Correlation Engine.
    - **Simulation mode**: Simulates network and processing latency when the
      engine is unavailable, useful for capacity planning and CI.
    """

    AGENT_COUNTS = [100, 1000, 5000, 10000]

    def __init__(
        self,
        endpoint: str,
        event_rate_hz: float = 1.0,
        sdc_probability: float = 0.001,
    ):
        self.endpoint = endpoint
        self.event_rate_hz = event_rate_hz
        self.sdc_probability = sdc_probability
        self._grpc_sender: GrpcSender | None = None
        self._mode = "simulated"

        self._try_connect()

    def _try_connect(self) -> None:
        """Attempt gRPC connection to the Correlation Engine."""
        try:
            sender = GrpcSender(self.endpoint)
            if sender._grpc_available:
                # Verify connection with a health check.
                import grpc
                try:
                    grpc.channel_ready_future(sender._channel).result(timeout=5)
                    self._grpc_sender = sender
                    self._mode = "grpc"
                    proto_status = "proto stubs" if sender._proto_available else "generic JSON"
                    print(f"  Connected to {self.endpoint} ({proto_status})")
                except grpc.FutureTimeoutError:
                    print(f"  [warn] Could not reach {self.endpoint}. Running in simulation mode.")
                    sender.close()
            else:
                print("  [warn] gRPC not available. Running in simulation mode.")
        except Exception as exc:
            print(f"  [warn] Connection failed: {exc}. Running in simulation mode.")

    def _send_event_grpc(self, event: dict[str, Any]) -> None:
        """Send event via gRPC to the real Correlation Engine."""
        if self._grpc_sender is None:
            raise RuntimeError("gRPC sender not initialized")
        self._grpc_sender.send_event(event)

    def _send_event_simulated(self, event: dict[str, Any]) -> None:
        """Simulate sending with realistic latency profile.

        Models a typical gRPC round-trip including serialization,
        network transit, server processing, and deserialization.
        """
        # Base latency: serialization + network + deserialization.
        base_latency = random.gauss(0.0015, 0.0004)
        # Server processing time (correlation, Bayesian update).
        processing = random.gauss(0.0008, 0.0002)
        # Occasional GC pause or batch flush (~1% of calls).
        if random.random() < 0.01:
            processing += random.uniform(0.005, 0.020)
        total = max(0.0001, base_latency + processing)
        time.sleep(total)

    def run_load_test(
        self,
        agent_count: int,
        duration_s: float = 60.0,
    ) -> LoadTestResult:
        """Run a load test with the specified number of agents."""
        print(f"\n  Load test: {agent_count} agents for {duration_s}s ({self._mode} mode)...")

        send_fn = self._send_event_grpc if self._mode == "grpc" else self._send_event_simulated

        # Create agents with unique GPU UUIDs.
        agents: list[AgentSimulator] = []
        for i in range(agent_count):
            node_id = f"node-{i // 8:04d}"
            gpu_uuid = f"GPU-{uuid.uuid5(uuid.NAMESPACE_DNS, f'gpu-{i}')}"
            agents.append(AgentSimulator(
                agent_id=f"agent-{i:05d}",
                gpu_uuid=gpu_uuid,
                node_id=node_id,
                event_rate_hz=self.event_rate_hz,
                sdc_probability=self.sdc_probability,
            ))

        # Run all agents concurrently.
        threads: list[threading.Thread] = []
        start_time = time.time()

        for agent in agents:
            t = threading.Thread(
                target=agent.run,
                args=(duration_s, send_fn),
                daemon=True,
            )
            threads.append(t)
            t.start()

        # Wait for completion.
        for t in threads:
            t.join(timeout=duration_s + 30)

        elapsed = time.time() - start_time

        # Aggregate results.
        all_latencies: list[float] = []
        total_sent = 0
        total_acked = 0
        total_errors = 0

        for agent in agents:
            all_latencies.extend(agent.send_latencies)
            total_sent += agent.events_sent
            total_acked += agent.events_acked
            total_errors += agent.errors

        if not all_latencies:
            all_latencies = [0.0]

        sorted_lat = sorted(all_latencies)
        n = len(sorted_lat)

        # Estimate correlation latency.
        # In gRPC mode, this is the observed round-trip time.
        # In simulation mode, model as 2-5x send latency (processing pipeline).
        if self._mode == "grpc":
            corr_latencies = sorted_lat  # Actual measured latency includes correlation.
        else:
            corr_latencies = [lat * random.uniform(2, 5) for lat in sorted_lat[:min(n, 1000)]]
        sorted_corr = sorted(corr_latencies)
        nc = len(sorted_corr)

        # Memory estimate: base + per-agent state + event buffer.
        memory_mb = 128 + agent_count * 0.5 + (total_sent * 0.001)

        # If running against real engine, try to get actual memory usage.
        if self._mode == "grpc" and self._grpc_sender is not None:
            try:
                import grpc
                # Query metrics endpoint for memory info.
                import urllib.request
                host = self.endpoint.split(":")[0]
                resp = urllib.request.urlopen(f"http://{host}:9090/metrics", timeout=3)
                for line in resp.read().decode().splitlines():
                    if line.startswith("process_resident_memory_bytes"):
                        memory_mb = float(line.split()[-1]) / (1024 * 1024)
                        break
            except Exception:
                pass  # Fall back to estimate.

        eps = total_sent / elapsed if elapsed > 0 else 0
        error_rate = (total_errors / total_sent * 100) if total_sent > 0 else 0

        result = LoadTestResult(
            agent_count=agent_count,
            duration_seconds=round(elapsed, 1),
            total_events_sent=total_sent,
            total_events_acked=total_acked,
            events_per_second=round(eps, 1),
            send_latency_p50_ms=round(sorted_lat[int(n * 0.50)], 3),
            send_latency_p95_ms=round(sorted_lat[int(n * 0.95)], 3),
            send_latency_p99_ms=round(sorted_lat[int(n * 0.99)], 3),
            send_latency_mean_ms=round(statistics.mean(sorted_lat), 3),
            correlation_latency_p50_ms=round(sorted_corr[int(nc * 0.50)], 3),
            correlation_latency_p95_ms=round(sorted_corr[int(nc * 0.95)], 3),
            correlation_latency_p99_ms=round(sorted_corr[int(nc * 0.99)], 3),
            errors=total_errors,
            error_rate_pct=round(error_rate, 2),
            memory_usage_mb=round(memory_mb, 1),
            mode=self._mode,
        )

        print(f"    Events/sec: {eps:.1f}")
        print(f"    Send latency p50/p95/p99: "
              f"{result.send_latency_p50_ms:.3f}/"
              f"{result.send_latency_p95_ms:.3f}/"
              f"{result.send_latency_p99_ms:.3f} ms")
        print(f"    Correlation latency p50/p95/p99: "
              f"{result.correlation_latency_p50_ms:.3f}/"
              f"{result.correlation_latency_p95_ms:.3f}/"
              f"{result.correlation_latency_p99_ms:.3f} ms")
        print(f"    Errors: {total_errors} ({error_rate:.2f}%)")
        print(f"    Memory (est): {result.memory_usage_mb:.1f} MB")

        return result

    def run_all(
        self,
        agent_counts: Optional[list[int]] = None,
        duration_s: float = 60.0,
    ) -> list[LoadTestResult]:
        """Run load tests at all agent counts."""
        if agent_counts is None:
            agent_counts = self.AGENT_COUNTS

        results: list[LoadTestResult] = []
        for count in agent_counts:
            result = self.run_load_test(count, duration_s)
            results.append(result)

        self._print_summary(results)
        return results

    def _print_summary(self, results: list[LoadTestResult]) -> None:
        print(f"\n{'='*110}")
        print(f"Correlation Engine Load Test Summary  (mode: {self._mode})")
        print(f"{'='*110}")
        print(
            f"{'Agents':>8} {'Events/s':>10} {'Send p50':>10} {'Send p95':>10} "
            f"{'Corr p50':>10} {'Corr p95':>10} {'Errors':>8} {'Err%':>7} {'Mem (MB)':>10}"
        )
        print("-" * 110)
        for r in results:
            print(
                f"{r.agent_count:>8} {r.events_per_second:>10.1f} "
                f"{r.send_latency_p50_ms:>10.3f} {r.send_latency_p95_ms:>10.3f} "
                f"{r.correlation_latency_p50_ms:>10.3f} "
                f"{r.correlation_latency_p95_ms:>10.3f} "
                f"{r.errors:>8} {r.error_rate_pct:>6.2f}% {r.memory_usage_mb:>10.1f}"
            )
        print(f"{'='*110}\n")

    def close(self) -> None:
        if self._grpc_sender:
            self._grpc_sender.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load test the SENTINEL Correlation Engine"
    )
    parser.add_argument("--endpoint", default="localhost:50051", help="gRPC endpoint")
    parser.add_argument(
        "--agents",
        help="Comma-separated agent counts (default: 100,1000,5000,10000)",
    )
    parser.add_argument("--duration", type=float, default=60, help="Duration per test (s)")
    parser.add_argument("--event-rate", type=float, default=1.0, help="Events/sec per agent")
    parser.add_argument("--sdc-probability", type=float, default=0.001,
                        help="Probability of SDC per event (0.0-1.0)")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    agent_counts = (
        [int(x) for x in args.agents.split(",")]
        if args.agents
        else CorrelationEngineLoadTest.AGENT_COUNTS
    )

    print(f"SENTINEL Correlation Engine Load Test")
    print(f"  Endpoint: {args.endpoint}")
    print(f"  Agent counts: {agent_counts}")
    print(f"  Duration: {args.duration}s per test")
    print(f"  Event rate: {args.event_rate} Hz per agent")
    print(f"  SDC probability: {args.sdc_probability}")

    lt = CorrelationEngineLoadTest(
        endpoint=args.endpoint,
        event_rate_hz=args.event_rate,
        sdc_probability=args.sdc_probability,
    )

    try:
        results = lt.run_all(agent_counts, args.duration)
    finally:
        lt.close()

    if args.output:
        output = {
            "endpoint": args.endpoint,
            "mode": lt._mode,
            "event_rate_hz": args.event_rate,
            "sdc_probability": args.sdc_probability,
            "results": [r.__dict__ for r in results],
        }
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
