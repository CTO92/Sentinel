//! Prometheus metrics for the SENTINEL Audit Ledger.

use lazy_static::lazy_static;
use prometheus::{
    Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, Opts, Registry,
};

lazy_static! {
    /// Shared Prometheus registry for the audit ledger.
    pub static ref REGISTRY: Registry = Registry::new_custom(
        Some("sentinel".into()),
        None,
    )
    .expect("failed to create Prometheus registry");

    /// Total number of audit entries ingested.
    pub static ref AUDIT_ENTRIES_TOTAL: IntCounter = {
        let c = IntCounter::with_opts(
            Opts::new("audit_entries_total", "Total audit entries ingested"),
        )
        .expect("metric creation");
        REGISTRY.register(Box::new(c.clone())).expect("register");
        c
    };

    /// Total number of batches committed.
    pub static ref AUDIT_BATCHES_TOTAL: IntCounter = {
        let c = IntCounter::with_opts(
            Opts::new("audit_batches_total", "Total audit batches committed"),
        )
        .expect("metric creation");
        REGISTRY.register(Box::new(c.clone())).expect("register");
        c
    };

    /// Chain verification outcomes.
    pub static ref CHAIN_VERIFICATION_RESULT: IntCounterVec = {
        let c = IntCounterVec::new(
            Opts::new("chain_verification_result", "Chain verification outcomes"),
            &["result"],
        )
        .expect("metric creation");
        REGISTRY.register(Box::new(c.clone())).expect("register");
        c
    };

    /// Query latency histogram (seconds).
    pub static ref QUERY_LATENCY_SECONDS: Histogram = {
        let h = Histogram::with_opts(
            HistogramOpts::new("query_latency_seconds", "Audit query latency in seconds")
                .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]),
        )
        .expect("metric creation");
        REGISTRY.register(Box::new(h.clone())).expect("register");
        h
    };

    /// Approximate storage usage in bytes (updated periodically).
    pub static ref AUDIT_STORAGE_BYTES: IntGauge = {
        let g = IntGauge::with_opts(
            Opts::new("audit_storage_bytes", "Approximate audit storage size in bytes"),
        )
        .expect("metric creation");
        REGISTRY.register(Box::new(g.clone())).expect("register");
        g
    };

    /// Total compliance reports generated.
    pub static ref COMPLIANCE_REPORTS_TOTAL: IntCounter = {
        let c = IntCounter::with_opts(
            Opts::new("compliance_reports_generated_total", "Total compliance reports generated"),
        )
        .expect("metric creation");
        REGISTRY.register(Box::new(c.clone())).expect("register");
        c
    };
}

/// Gather all metrics as a Prometheus text exposition string.
pub fn gather_metrics() -> String {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .expect("encode metrics");
    String::from_utf8(buffer).expect("metrics are valid UTF-8")
}
