//! Benchmark for event processing throughput in the correlation pipeline.

use std::collections::HashMap;
use std::time::Duration;

use chrono::Utc;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use sentinel_correlation_engine::correlation::bayesian_attribution::BayesianAttributor;
use sentinel_correlation_engine::correlation::pattern_matcher::{PatternMatcher, PatternMatcherConfig};
use sentinel_correlation_engine::correlation::temporal_window::{
    CorrelationEvent, EventType, TemporalWindow,
};
use sentinel_correlation_engine::util::config::BayesianConfig;

fn make_event(gpu: &str, event_type: EventType) -> CorrelationEvent {
    CorrelationEvent {
        event_id: uuid::Uuid::new_v4().to_string(),
        gpu_uuid: gpu.to_string(),
        sm_id: Some(0),
        event_type,
        timestamp: Utc::now(),
        severity: 2,
        score: 1.0,
        metadata: HashMap::new(),
    }
}

fn bench_temporal_window_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_window_insert");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let mut window = TemporalWindow::new(Duration::from_secs(300));
                for _ in 0..size {
                    window.insert(make_event("gpu-1", EventType::ProbePass));
                }
            });
        });
    }

    group.finish();
}

fn bench_temporal_window_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_window_query");

    for size in [100, 1_000, 10_000].iter() {
        // Pre-populate the window.
        let mut window = TemporalWindow::new(Duration::from_secs(300));
        for _ in 0..*size {
            window.insert(make_event("gpu-1", EventType::ProbePass));
        }
        // Add some failures for variety.
        for _ in 0..(*size / 10) {
            window.insert(make_event("gpu-1", EventType::ProbeFail));
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let events = window.events_for_gpu("gpu-1");
                criterion::black_box(events.len());
            });
        });
    }

    group.finish();
}

fn bench_pattern_matching(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");
    let matcher = PatternMatcher::with_defaults();

    for size in [10, 50, 100].iter() {
        let events: Vec<CorrelationEvent> = (0..*size)
            .map(|i| {
                if i % 3 == 0 {
                    make_event("gpu-1", EventType::ProbeFail)
                } else if i % 3 == 1 {
                    make_event("gpu-1", EventType::InferenceAnomaly)
                } else {
                    make_event("gpu-1", EventType::ProbePass)
                }
            })
            .collect();
        let event_refs: Vec<&CorrelationEvent> = events.iter().collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let patterns = matcher.match_patterns("gpu-1", &event_refs);
                criterion::black_box(patterns.len());
            });
        });
    }

    group.finish();
}

fn bench_bayesian_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("bayesian_update");
    let attributor = BayesianAttributor::new(BayesianConfig::default());

    group.bench_function("probe_success", |b| {
        b.iter(|| {
            attributor.record_probe_success("gpu-bench", Some(0));
        });
    });

    group.bench_function("probe_failure", |b| {
        b.iter(|| {
            attributor.record_probe_failure("gpu-bench", Some(0));
        });
    });

    group.bench_function("anomaly", |b| {
        b.iter(|| {
            attributor.record_anomaly("gpu-bench");
        });
    });

    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    let attributor = BayesianAttributor::new(BayesianConfig::default());
    let matcher = PatternMatcher::with_defaults();

    group.throughput(Throughput::Elements(1));
    group.bench_function("process_event", |b| {
        let mut window = TemporalWindow::new(Duration::from_secs(300));

        b.iter(|| {
            let event = make_event("gpu-1", EventType::ProbeFail);

            // Step 1: Add to temporal window.
            window.insert(event.clone());

            // Step 2: Update Bayesian belief.
            attributor.record_probe_failure("gpu-1", Some(0));

            // Step 3: Run pattern matcher.
            let events = window.events_for_gpu("gpu-1");
            let patterns = matcher.match_patterns("gpu-1", &events);

            criterion::black_box(patterns.len());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_temporal_window_insert,
    bench_temporal_window_query,
    bench_pattern_matching,
    bench_bayesian_update,
    bench_full_pipeline,
);
criterion_main!(benches);
