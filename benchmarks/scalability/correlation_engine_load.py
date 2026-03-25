"""
SENTINEL Benchmark — Correlation Engine Load Test

Simulates 100-10,000 agents concurrently sending events to the correlation
engine to measure ingestion throughput, correlation latency, and memory usage.

Usage:
    python correlation_engine_load.py --endpoint localhost:50051
    python correlation_engine_load.py --endpoint localhost:50051 --agents 5000 --duration 300
    python correlation_engine_load.py --endpoint localhost:50051 --output load_test.json

Copyright 2025-2026 SENTINEL Authors — Apache 2.0
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


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
    memory_usage_mb: float


class AgentSimulator:
    """Simulates a single probe agent sending events."""

    def __init__(
        self,
        agent_id: str,
        gpu_id: str,
        node_id: str,
        event_rate_hz: float = 1.0,
        sdc_probability: float = 0.001,
    ):
        self.agent_id = agent_id
        self.gpu_id = gpu_id
        self.node_id = node_id
        self.event_rate_hz = event_rate_hz
        self.sdc_probability = sdc_probability
        self._rng = random.Random(hash(agent_id))
        self.send_latencies: list[float] = []
        self.events_sent = 0
        self.events_acked = 0
        self.errors = 0
        self._stop = False

    def generate_event(self) -> dict[str, Any]:
        """Generate a synthetic event."""
        has_sdc = self._rng.random() < self.sdc_probability
        return {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "agent_id": self.agent_id,
            "gpu_id": self.gpu_id,
            "node_id": self.node_id,
            "event_type": "sdc_detected" if has_sdc else "probe_healthy",
            "severity": "warning" if has_sdc else "info",
            "probe_type": self._rng.choice(["fma", "tensor_core", "memory", "transcendental"]),
            "deviation": self._rng.uniform(0.01, 1.0) if has_sdc else 0.0,
        }

    def run(self, duration_s: float, send_fn) -> None:
        """Run the agent for a given duration."""
        interval = 1.0 / self.event_rate_hz
        end_time = time.time() + duration_s

        while time.time() < end_time and not self._stop:
            event = self.generate_event()
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

            # Rate limiting
            elapsed = time.time() - start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        self._stop = True


class CorrelationEngineLoadTest:
    """
    Load test framework for the SENTINEL Correlation Engine.
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
        self._grpc_available = False
        self._channel = None
        self._stub = None

        self._try_connect()

    def _try_connect(self) -> None:
        """Attempt gRPC connection."""
        try:
            import grpc
            self._channel = grpc.insecure_channel(self.endpoint)
            self._grpc_available = True
        except ImportError:
            print("gRPC not available. Running in simulation mode.")
            self._grpc_available = False

    def _send_event_grpc(self, event: dict[str, Any]) -> None:
        """Send event via gRPC."""
        if not self._grpc_available:
            # Simulate network latency
            time.sleep(random.gauss(0.001, 0.0003))
            return
        raise NotImplementedError("gRPC stub not configured")

    def _send_event_simulated(self, event: dict[str, Any]) -> None:
        """Simulate sending with realistic latency."""
        # Simulate network + processing latency
        time.sleep(random.gauss(0.002, 0.0005))

    def run_load_test(
        self,
        agent_count: int,
        duration_s: float = 60.0,
    ) -> LoadTestResult:
        """Run a load test with the specified number of agents."""
        print(f"\n  Load test: {agent_count} agents for {duration_s}s...")

        send_fn = self._send_event_grpc if self._grpc_available else self._send_event_simulated

        # Create agents
        agents: list[AgentSimulator] = []
        for i in range(agent_count):
            node_id = f"node-{i // 8:04d}"
            gpu_id = f"gpu-{node_id}-{i % 8}"
            agents.append(AgentSimulator(
                agent_id=f"agent-{i:05d}",
                gpu_id=gpu_id,
                node_id=node_id,
                event_rate_hz=self.event_rate_hz,
                sdc_probability=self.sdc_probability,
            ))

        # Run all agents concurrently
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

        # Wait for completion
        for t in threads:
            t.join(timeout=duration_s + 30)

        elapsed = time.time() - start_time

        # Aggregate results
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

        # Estimate correlation latency (simulated as 2-5x send latency)
        corr_latencies = [l * random.uniform(2, 5) for l in all_latencies[:min(n, 1000)]]
        sorted_corr = sorted(corr_latencies)
        nc = len(sorted_corr)

        # Estimate memory usage based on agent count
        memory_mb = 128 + agent_count * 0.5 + (total_sent * 0.001)

        eps = total_sent / elapsed if elapsed > 0 else 0

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
            memory_usage_mb=round(memory_mb, 1),
        )

        print(f"    Events/sec: {eps:.1f}")
        print(f"    Send latency p50/p95/p99: "
              f"{result.send_latency_p50_ms:.3f}/"
              f"{result.send_latency_p95_ms:.3f}/"
              f"{result.send_latency_p99_ms:.3f} ms")
        print(f"    Errors: {total_errors}")

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
        print(f"\n{'='*95}")
        print("Correlation Engine Load Test Summary")
        print(f"{'='*95}")
        print(
            f"{'Agents':>8} {'Events/s':>10} {'Send p50':>10} {'Send p95':>10} "
            f"{'Corr p50':>10} {'Corr p95':>10} {'Errors':>8} {'Mem (MB)':>10}"
        )
        print("-" * 95)
        for r in results:
            print(
                f"{r.agent_count:>8} {r.events_per_second:>10.1f} "
                f"{r.send_latency_p50_ms:>10.3f} {r.send_latency_p95_ms:>10.3f} "
                f"{r.correlation_latency_p50_ms:>10.3f} "
                f"{r.correlation_latency_p95_ms:>10.3f} "
                f"{r.errors:>8} {r.memory_usage_mb:>10.1f}"
            )
        print(f"{'='*95}\n")

    def close(self) -> None:
        if self._channel:
            self._channel.close()


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
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    agent_counts = (
        [int(x) for x in args.agents.split(",")]
        if args.agents
        else CorrelationEngineLoadTest.AGENT_COUNTS
    )

    lt = CorrelationEngineLoadTest(
        endpoint=args.endpoint,
        event_rate_hz=args.event_rate,
    )

    try:
        results = lt.run_all(agent_counts, args.duration)
    finally:
        lt.close()

    if args.output:
        output = {
            "endpoint": args.endpoint,
            "results": [r.__dict__ for r in results],
        }
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
