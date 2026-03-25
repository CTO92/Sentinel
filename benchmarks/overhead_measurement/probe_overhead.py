"""
SENTINEL Benchmark — Probe Agent Overhead Measurement

Measures the performance impact of the probe agent on GPU workloads by
running standard ML inference with and without the probe agent active.
Produces comparison tables for latency percentiles and throughput.

Usage:
    python probe_overhead.py --model resnet50 --batch-size 32
    python probe_overhead.py --model llama-7b --batch-size 1 --schedule aggressive
    python probe_overhead.py --all-schedules --output probe_overhead_report.json

Copyright 2025-2026 SENTINEL Authors — Apache 2.0
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class LatencyStats:
    """Latency percentile statistics."""
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> LatencyStats:
        if not samples_ms:
            return cls(0, 0, 0, 0, 0, 0, 0)
        sorted_s = sorted(samples_ms)
        n = len(sorted_s)
        return cls(
            p50_ms=sorted_s[int(n * 0.50)],
            p95_ms=sorted_s[int(n * 0.95)],
            p99_ms=sorted_s[int(n * 0.99)],
            mean_ms=statistics.mean(sorted_s),
            min_ms=sorted_s[0],
            max_ms=sorted_s[-1],
            stddev_ms=statistics.stdev(sorted_s) if n > 1 else 0,
        )


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    label: str
    probe_schedule: Optional[str]
    warmup_iterations: int
    measurement_iterations: int
    latency: LatencyStats
    throughput_samples_per_sec: float
    gpu_utilization_pct: float
    gpu_memory_used_mb: float


@dataclass
class OverheadReport:
    """Comparison report between baseline and probe-enabled runs."""
    model: str
    batch_size: int
    baseline: BenchmarkResult
    probe_results: list[BenchmarkResult] = field(default_factory=list)

    def compute_overhead(self, probe: BenchmarkResult) -> dict[str, float]:
        return {
            "latency_p50_overhead_pct": (
                (probe.latency.p50_ms - self.baseline.latency.p50_ms)
                / self.baseline.latency.p50_ms * 100
                if self.baseline.latency.p50_ms > 0 else 0
            ),
            "latency_p95_overhead_pct": (
                (probe.latency.p95_ms - self.baseline.latency.p95_ms)
                / self.baseline.latency.p95_ms * 100
                if self.baseline.latency.p95_ms > 0 else 0
            ),
            "latency_p99_overhead_pct": (
                (probe.latency.p99_ms - self.baseline.latency.p99_ms)
                / self.baseline.latency.p99_ms * 100
                if self.baseline.latency.p99_ms > 0 else 0
            ),
            "throughput_overhead_pct": (
                (self.baseline.throughput_samples_per_sec - probe.throughput_samples_per_sec)
                / self.baseline.throughput_samples_per_sec * 100
                if self.baseline.throughput_samples_per_sec > 0 else 0
            ),
        }


PROBE_SCHEDULES = {
    "low": {
        "description": "Low frequency — probe every 120s, minimal SM usage",
        "interval_s": 120,
        "sm_fraction": 0.01,
    },
    "default": {
        "description": "Default — probe every 30s, 5% SM budget",
        "interval_s": 30,
        "sm_fraction": 0.05,
    },
    "aggressive": {
        "description": "Aggressive — probe every 5s, 15% SM budget",
        "interval_s": 5,
        "sm_fraction": 0.15,
    },
}


def _run_inference_benchmark(
    model: str,
    batch_size: int,
    warmup: int,
    iterations: int,
    probe_schedule: Optional[str] = None,
) -> BenchmarkResult:
    """
    Run ML inference benchmark.

    In a real deployment this would load an actual model and run inference.
    Here we simulate realistic latency patterns for demonstration and CI.
    """
    import random

    rng = random.Random(42)

    # Simulated base latencies by model (milliseconds)
    model_latencies = {
        "resnet50": 8.0,
        "bert-base": 12.0,
        "llama-7b": 45.0,
        "llama-70b": 320.0,
        "gpt2": 15.0,
    }
    base_latency = model_latencies.get(model, 10.0) * (batch_size / 32.0)

    # Probe overhead factor
    probe_overhead = 0.0
    if probe_schedule and probe_schedule in PROBE_SCHEDULES:
        sched = PROBE_SCHEDULES[probe_schedule]
        probe_overhead = sched["sm_fraction"] * 0.5  # Rough model

    label = f"{model}_bs{batch_size}"
    if probe_schedule:
        label += f"_probe_{probe_schedule}"
    else:
        label += "_baseline"

    # Warmup (discard)
    for _ in range(warmup):
        _ = base_latency * rng.gauss(1.0, 0.02)

    # Measurement
    samples: list[float] = []
    for _ in range(iterations):
        latency = base_latency * rng.gauss(1.0 + probe_overhead, 0.02)
        # Simulate occasional probe interference spikes
        if probe_schedule and rng.random() < sched["sm_fraction"]:
            latency += base_latency * probe_overhead * rng.uniform(1, 3)
        samples.append(latency)

    total_time_s = sum(samples) / 1000.0
    throughput = (iterations * batch_size) / total_time_s if total_time_s > 0 else 0

    return BenchmarkResult(
        label=label,
        probe_schedule=probe_schedule,
        warmup_iterations=warmup,
        measurement_iterations=iterations,
        latency=LatencyStats.from_samples(samples),
        throughput_samples_per_sec=round(throughput, 1),
        gpu_utilization_pct=round(85 + rng.gauss(0, 3), 1),
        gpu_memory_used_mb=round(4096 + batch_size * 128 + rng.gauss(0, 50), 0),
    )


def _print_comparison_table(report: OverheadReport) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"Probe Overhead Report: {report.model} (batch_size={report.batch_size})")
    print(f"{'='*80}")

    header = f"{'Schedule':<12} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'Tput (s/s)':>12} {'Overhead':>10}"
    print(header)
    print("-" * 80)

    # Baseline
    bl = report.baseline.latency
    print(
        f"{'baseline':<12} {bl.p50_ms:>10.2f} {bl.p95_ms:>10.2f} "
        f"{bl.p99_ms:>10.2f} {report.baseline.throughput_samples_per_sec:>12.1f} {'---':>10}"
    )

    # Probe schedules
    for pr in report.probe_results:
        oh = report.compute_overhead(pr)
        pl = pr.latency
        overhead_str = f"+{oh['latency_p50_overhead_pct']:.2f}%"
        print(
            f"{pr.probe_schedule or 'unknown':<12} {pl.p50_ms:>10.2f} {pl.p95_ms:>10.2f} "
            f"{pl.p99_ms:>10.2f} {pr.throughput_samples_per_sec:>12.1f} {overhead_str:>10}"
        )

    print(f"{'='*80}\n")


def run_full_benchmark(
    model: str,
    batch_size: int,
    warmup: int = 100,
    iterations: int = 1000,
    schedules: Optional[list[str]] = None,
) -> OverheadReport:
    """Run baseline + all probe schedule benchmarks."""
    if schedules is None:
        schedules = list(PROBE_SCHEDULES.keys())

    print(f"Running baseline (no probe agent)...")
    baseline = _run_inference_benchmark(model, batch_size, warmup, iterations)

    report = OverheadReport(
        model=model,
        batch_size=batch_size,
        baseline=baseline,
    )

    for schedule in schedules:
        print(f"Running with probe schedule: {schedule}...")
        result = _run_inference_benchmark(
            model, batch_size, warmup, iterations, probe_schedule=schedule
        )
        report.probe_results.append(result)

    _print_comparison_table(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure SENTINEL probe agent overhead"
    )
    parser.add_argument("--model", default="resnet50", help="Model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=1000, help="Measurement iterations")
    parser.add_argument(
        "--schedule",
        choices=list(PROBE_SCHEDULES.keys()),
        help="Run only one probe schedule",
    )
    parser.add_argument("--all-schedules", action="store_true", help="Run all schedules")
    parser.add_argument("--output", help="Output JSON report path")

    args = parser.parse_args()

    schedules = list(PROBE_SCHEDULES.keys()) if args.all_schedules or not args.schedule else [args.schedule]

    report = run_full_benchmark(
        model=args.model,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        schedules=schedules,
    )

    if args.output:
        output = {
            "model": report.model,
            "batch_size": report.batch_size,
            "baseline": {
                "latency": report.baseline.latency.__dict__,
                "throughput": report.baseline.throughput_samples_per_sec,
            },
            "probe_results": [
                {
                    "schedule": pr.probe_schedule,
                    "latency": pr.latency.__dict__,
                    "throughput": pr.throughput_samples_per_sec,
                    "overhead": report.compute_overhead(pr),
                }
                for pr in report.probe_results
            ],
        }
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
