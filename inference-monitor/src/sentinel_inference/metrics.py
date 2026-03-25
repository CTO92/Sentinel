"""Prometheus metrics for the SENTINEL Inference Monitor.

Exposes counters, histograms, and gauges for observability of the
sidecar's sampling, analysis, and reporting pipeline.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
SAMPLES_TOTAL = Counter(
    "sentinel_inference_samples_total",
    "Total number of inference output tensors sampled.",
    ["node_id", "gpu_id"],
)

ANOMALIES_TOTAL = Counter(
    "sentinel_inference_anomalies_total",
    "Total anomaly events raised, by type.",
    ["node_id", "gpu_id", "type"],
)

GRPC_ERRORS_TOTAL = Counter(
    "sentinel_inference_grpc_errors_total",
    "Total gRPC reporting errors.",
    ["node_id", "gpu_id", "error_type"],
)

DROPPED_SAMPLES_TOTAL = Counter(
    "sentinel_inference_dropped_samples_total",
    "Samples dropped due to full queue.",
    ["node_id", "gpu_id"],
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------
ANALYSIS_DURATION = Histogram(
    "sentinel_inference_analysis_duration_seconds",
    "Time spent analyzing a single tensor sample.",
    ["node_id", "gpu_id", "analyzer"],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)

INTERCEPT_DURATION = Histogram(
    "sentinel_inference_intercept_duration_seconds",
    "Time spent capturing a tensor from the inference server.",
    ["node_id", "gpu_id"],
    buckets=(0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01),
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------
QUEUE_SIZE = Gauge(
    "sentinel_inference_queue_size",
    "Current number of samples pending analysis.",
    ["node_id", "gpu_id"],
)

EWMA_VALUE = Gauge(
    "sentinel_inference_ewma_value",
    "Current EWMA value for a tracked statistic.",
    ["node_id", "gpu_id", "stat"],
)


def start_metrics_server(host: str = "0.0.0.0", port: int = 9090) -> None:
    """Start the Prometheus HTTP metrics server in a daemon thread."""
    start_http_server(port, addr=host)
