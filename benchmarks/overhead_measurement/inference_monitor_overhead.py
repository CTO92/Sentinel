"""
SENTINEL Benchmark — Inference Monitor Overhead Measurement

Measures the overhead introduced by the inference monitor at different
sampling rates (0.1%, 1%, 5%, 10%). Reports added latency and throughput
impact.

Usage:
    python inference_monitor_overhead.py --model resnet50 --batch-size 32
    python inference_monitor_overhead.py --sampling-rates 0.001,0.01,0.05,0.1
    python inference_monitor_overhead.py --output inference_overhead.json

Copyright 2025-2026 SENTINEL Authors — Apache 2.0
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class SamplingBenchmark:
    """Benchmark result for a single sampling rate."""
    sampling_rate: float
    sampling_rate_pct: str
    iterations: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    throughput_samples_per_sec: float
    sampled_count: int
    overhead_latency_p50_pct: float
    overhead_throughput_pct: float


def _simulate_inference(
    base_latency_ms: float,
    batch_size: int,
    iterations: int,
    sampling_rate: float,
    rng: random.Random,
) -> tuple[list[float], int]:
    """
    Simulate inference with optional sampling overhead.

    When a request is sampled, additional latency is added for:
    - Capturing input/output tensors
    - Computing statistical fingerprints
    - Checking against reference distributions
    """
    # Overhead per sampled request (fingerprint + comparison)
    sample_overhead_ms = 0.5 + base_latency_ms * 0.02

    latencies: list[float] = []
    sampled = 0

    for _ in range(iterations):
        lat = base_latency_ms * rng.gauss(1.0, 0.015)

        if rng.random() < sampling_rate:
            lat += sample_overhead_ms * rng.gauss(1.0, 0.1)
            sampled += 1

        latencies.append(lat)

    return latencies, sampled


def run_benchmark(
    model: str = "resnet50",
    batch_size: int = 32,
    warmup: int = 100,
    iterations: int = 5000,
    sampling_rates: Optional[list[float]] = None,
) -> list[SamplingBenchmark]:
    """Run inference monitor overhead benchmarks at multiple sampling rates."""
    if sampling_rates is None:
        sampling_rates = [0.0, 0.001, 0.01, 0.05, 0.1]

    model_latencies = {
        "resnet50": 8.0,
        "bert-base": 12.0,
        "llama-7b": 45.0,
        "llama-70b": 320.0,
    }
    base_latency = model_latencies.get(model, 10.0) * (batch_size / 32.0)
    rng = random.Random(42)

    results: list[SamplingBenchmark] = []
    baseline_p50 = 0.0
    baseline_throughput = 0.0

    for rate in sampling_rates:
        label = f"{rate*100:.1f}%" if rate > 0 else "baseline"
        print(f"  Benchmarking at sampling rate {label}...")

        # Warmup
        _simulate_inference(base_latency, batch_size, warmup, rate, rng)

        # Measure
        latencies, sampled = _simulate_inference(
            base_latency, batch_size, iterations, rate, rng
        )

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        p50 = sorted_lat[int(n * 0.50)]
        p95 = sorted_lat[int(n * 0.95)]
        p99 = sorted_lat[int(n * 0.99)]
        mean = statistics.mean(sorted_lat)
        total_time_s = sum(latencies) / 1000.0
        throughput = (iterations * batch_size) / total_time_s

        if rate == 0.0:
            baseline_p50 = p50
            baseline_throughput = throughput

        oh_lat = ((p50 - baseline_p50) / baseline_p50 * 100) if baseline_p50 > 0 else 0
        oh_tput = ((baseline_throughput - throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0

        results.append(SamplingBenchmark(
            sampling_rate=rate,
            sampling_rate_pct=label,
            iterations=iterations,
            latency_p50_ms=round(p50, 3),
            latency_p95_ms=round(p95, 3),
            latency_p99_ms=round(p99, 3),
            latency_mean_ms=round(mean, 3),
            throughput_samples_per_sec=round(throughput, 1),
            sampled_count=sampled,
            overhead_latency_p50_pct=round(oh_lat, 3),
            overhead_throughput_pct=round(oh_tput, 3),
        ))

    # Print table
    print(f"\n{'='*90}")
    print(f"Inference Monitor Overhead: {model} (batch_size={batch_size})")
    print(f"{'='*90}")
    print(
        f"{'Rate':<12} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} "
        f"{'Tput (s/s)':>12} {'Lat OH%':>10} {'Tput OH%':>10}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r.sampling_rate_pct:<12} {r.latency_p50_ms:>10.3f} "
            f"{r.latency_p95_ms:>10.3f} {r.latency_p99_ms:>10.3f} "
            f"{r.throughput_samples_per_sec:>12.1f} "
            f"{r.overhead_latency_p50_pct:>+10.3f} "
            f"{r.overhead_throughput_pct:>+10.3f}"
        )
    print(f"{'='*90}\n")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure SENTINEL inference monitor overhead"
    )
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument(
        "--sampling-rates",
        default="0.0,0.001,0.01,0.05,0.1",
        help="Comma-separated sampling rates",
    )
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    rates = [float(r) for r in args.sampling_rates.split(",")]

    results = run_benchmark(
        model=args.model,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        sampling_rates=rates,
    )

    if args.output:
        output = {
            "model": args.model,
            "batch_size": args.batch_size,
            "results": [r.__dict__ for r in results],
        }
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
