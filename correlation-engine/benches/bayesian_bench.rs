//! Benchmark for Bayesian belief update computations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use sentinel_correlation_engine::correlation::bayesian_attribution::{
    BayesianAttributor, BayesianBelief,
};
use sentinel_correlation_engine::util::config::BayesianConfig;

fn bench_belief_update_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("belief_update");

    group.bench_function("success_update", |b| {
        let attributor = BayesianAttributor::new(BayesianConfig::default());
        b.iter(|| {
            attributor.record_probe_success("gpu-1", Some(0));
        });
    });

    group.bench_function("failure_update", |b| {
        let attributor = BayesianAttributor::new(BayesianConfig::default());
        b.iter(|| {
            attributor.record_probe_failure("gpu-1", Some(0));
        });
    });

    group.bench_function("anomaly_update", |b| {
        let attributor = BayesianAttributor::new(BayesianConfig::default());
        b.iter(|| {
            attributor.record_anomaly("gpu-1");
        });
    });

    group.finish();
}

fn bench_reliability_score(c: &mut Criterion) {
    let mut group = c.benchmark_group("reliability_score");

    let belief = BayesianBelief::new(1000.0, 1.0);

    group.bench_function("compute_score", |b| {
        b.iter(|| {
            criterion::black_box(belief.reliability_score());
        });
    });

    group.bench_function("compute_variance", |b| {
        b.iter(|| {
            criterion::black_box(belief.variance());
        });
    });

    group.bench_function("compute_credible_bound", |b| {
        b.iter(|| {
            criterion::black_box(belief.lower_credible_bound(0.95).unwrap());
        });
    });

    group.finish();
}

fn bench_multi_gpu_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_gpu");

    for gpu_count in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*gpu_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(gpu_count),
            gpu_count,
            |b, &count| {
                let attributor = BayesianAttributor::new(BayesianConfig::default());
                // Pre-register GPUs.
                for i in 0..count {
                    attributor.record_probe_success(&format!("gpu-{}", i), None);
                }

                b.iter(|| {
                    for i in 0..count {
                        attributor.record_probe_success(&format!("gpu-{}", i), None);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_tier_determination(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier_determination");
    let attributor = BayesianAttributor::new(BayesianConfig::default());

    group.bench_function("score_to_tier", |b| {
        b.iter(|| {
            criterion::black_box(attributor.score_to_tier(0.997));
        });
    });

    group.bench_function("gpu_tier_lookup", |b| {
        attributor.record_probe_success("gpu-1", None);
        b.iter(|| {
            criterion::black_box(attributor.gpu_tier("gpu-1"));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_belief_update_single,
    bench_reliability_score,
    bench_multi_gpu_updates,
    bench_tier_determination,
);
criterion_main!(benches);
