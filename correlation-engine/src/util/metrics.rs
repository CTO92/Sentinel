//! Prometheus metrics definitions for the Correlation Engine.
//!
//! All metrics use the `sentinel_` prefix for easy identification in
//! multi-service environments. Metrics are registered lazily on first access.

use once_cell::sync::Lazy;
use prometheus::{
    self, CounterVec, Encoder, GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec,
    Opts, Registry, TextEncoder,
};

/// Global metrics registry for the correlation engine.
static REGISTRY: Lazy<Registry> = Lazy::new(|| Registry::new_custom(None, None).unwrap());

/// Total number of events ingested, labeled by source (probe, anomaly, telemetry).
static EVENTS_INGESTED: Lazy<IntCounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "sentinel_events_ingested_total",
        "Total number of events ingested by the correlation engine",
    )
    .namespace("sentinel");
    let counter = IntCounterVec::new(opts, &["source"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

/// Histogram of correlation pipeline latency in seconds.
static CORRELATION_LATENCY: Lazy<HistogramVec> = Lazy::new(|| {
    let opts = HistogramOpts::new(
        "sentinel_correlation_latency_seconds",
        "Latency of the correlation pipeline per event",
    )
    .namespace("sentinel")
    .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]);
    let hist = HistogramVec::new(opts, &["event_type"]).unwrap();
    REGISTRY.register(Box::new(hist.clone())).unwrap();
    hist
});

/// Total number of quarantine actions, labeled by action type.
static QUARANTINE_ACTIONS: Lazy<IntCounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "sentinel_quarantine_actions_total",
        "Total quarantine actions taken",
    )
    .namespace("sentinel");
    let counter = IntCounterVec::new(opts, &["action"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

/// Current GPU health reliability score, labeled by GPU UUID.
static GPU_HEALTH_SCORE: Lazy<GaugeVec> = Lazy::new(|| {
    let opts = Opts::new(
        "sentinel_gpu_health_score",
        "Current Bayesian reliability score per GPU",
    )
    .namespace("sentinel");
    let gauge = GaugeVec::new(opts, &["gpu"]).unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

/// Trust graph coverage percentage (0.0 - 100.0).
static TRUST_GRAPH_COVERAGE: Lazy<prometheus::Gauge> = Lazy::new(|| {
    let opts = Opts::new(
        "sentinel_trust_graph_coverage_percent",
        "Percentage of GPU pairs with at least one trust comparison",
    )
    .namespace("sentinel");
    let gauge = prometheus::Gauge::with_opts(opts).unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

/// Total alerts sent, labeled by channel.
static ALERTS_SENT: Lazy<IntCounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "sentinel_alert_sent_total",
        "Total alerts dispatched by channel",
    )
    .namespace("sentinel");
    let counter = IntCounterVec::new(opts, &["channel"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

/// Number of GPUs per health state.
static GPU_STATE_GAUGE: Lazy<IntGaugeVec> = Lazy::new(|| {
    let opts = Opts::new(
        "sentinel_gpu_state_count",
        "Number of GPUs in each health state",
    )
    .namespace("sentinel");
    let gauge = IntGaugeVec::new(opts, &["state"]).unwrap();
    REGISTRY.register(Box::new(gauge.clone())).unwrap();
    gauge
});

/// Number of active correlation patterns detected.
static PATTERNS_DETECTED: Lazy<IntCounterVec> = Lazy::new(|| {
    let opts = Opts::new(
        "sentinel_patterns_detected_total",
        "Total correlation patterns detected",
    )
    .namespace("sentinel");
    let counter = IntCounterVec::new(opts, &["pattern_type"]).unwrap();
    REGISTRY.register(Box::new(counter.clone())).unwrap();
    counter
});

// ---------------------------------------------------------------------------
// Public accessor functions
// ---------------------------------------------------------------------------

/// Record an ingested event. `source` should be "probe", "anomaly", or "telemetry".
pub fn record_event_ingested(source: &str) {
    EVENTS_INGESTED.with_label_values(&[source]).inc();
}

/// Observe a correlation pipeline latency sample.
pub fn observe_correlation_latency(event_type: &str, seconds: f64) {
    CORRELATION_LATENCY
        .with_label_values(&[event_type])
        .observe(seconds);
}

/// Record a quarantine action. `action` should be "quarantine", "reinstate",
/// "condemn", or "deep_test".
pub fn record_quarantine_action(action: &str) {
    QUARANTINE_ACTIONS.with_label_values(&[action]).inc();
}

/// Update the health score gauge for a GPU.
pub fn set_gpu_health_score(gpu_uuid: &str, score: f64) {
    GPU_HEALTH_SCORE
        .with_label_values(&[gpu_uuid])
        .set(score);
}

/// Update trust graph coverage percentage.
pub fn set_trust_graph_coverage(pct: f64) {
    TRUST_GRAPH_COVERAGE.set(pct);
}

/// Record an alert sent on a given channel.
pub fn record_alert_sent(channel: &str) {
    ALERTS_SENT.with_label_values(&[channel]).inc();
}

/// Update the count of GPUs in a given state.
pub fn set_gpu_state_count(state: &str, count: i64) {
    GPU_STATE_GAUGE.with_label_values(&[state]).set(count);
}

/// Record a detected correlation pattern.
pub fn record_pattern_detected(pattern_type: &str) {
    PATTERNS_DETECTED
        .with_label_values(&[pattern_type])
        .inc();
}

/// Render all registered metrics in Prometheus text exposition format.
pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Initialize all metrics (forces lazy statics). Call once at startup.
pub fn init_metrics() {
    // Touch each lazy static so they register with the global registry.
    Lazy::force(&EVENTS_INGESTED);
    Lazy::force(&CORRELATION_LATENCY);
    Lazy::force(&QUARANTINE_ACTIONS);
    Lazy::force(&GPU_HEALTH_SCORE);
    Lazy::force(&TRUST_GRAPH_COVERAGE);
    Lazy::force(&ALERTS_SENT);
    Lazy::force(&GPU_STATE_GAUGE);
    Lazy::force(&PATTERNS_DETECTED);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_gather() {
        init_metrics();
        record_event_ingested("probe");
        record_event_ingested("anomaly");
        let output = gather_metrics();
        assert!(output.contains("sentinel_events_ingested_total"));
    }
}
