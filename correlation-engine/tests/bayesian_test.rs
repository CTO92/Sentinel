//! Tests for the Bayesian attribution model.
//!
//! Validates Beta distribution updates, threshold transitions, SM-level
//! tracking, and credible interval computations.

use sentinel_correlation_engine::correlation::bayesian_attribution::{
    BayesianAttributor, BayesianBelief, ReliabilityTier,
};
use sentinel_correlation_engine::util::config::BayesianConfig;

fn default_config() -> BayesianConfig {
    BayesianConfig::default()
}

#[test]
fn test_prior_reliability_is_near_one() {
    let belief = BayesianBelief::new(1000.0, 1.0);
    let score = belief.reliability_score();
    assert!(score > 0.999, "Expected > 0.999, got {}", score);
}

#[test]
fn test_single_failure_minimal_impact() {
    let attributor = BayesianAttributor::new(default_config());
    let tier = attributor.record_probe_failure("gpu-1", None);
    let score = attributor.gpu_reliability_score("gpu-1").unwrap();

    // With prior Beta(1000, 1), one failure -> Beta(1000, 2)
    // score = 1000 / 1002 = 0.998004...
    assert!(score > 0.998, "Score too low after single failure: {}", score);
    assert!(
        tier == ReliabilityTier::Healthy || tier == ReliabilityTier::IncreasedProbing,
        "Expected Healthy or IncreasedProbing, got {:?}",
        tier
    );
}

#[test]
fn test_multiple_failures_degrade_trust() {
    let attributor = BayesianAttributor::new(default_config());

    // Track reliability through successive failures.
    let mut scores = Vec::new();
    for i in 0..20 {
        attributor.record_probe_failure("gpu-1", None);
        let score = attributor.gpu_reliability_score("gpu-1").unwrap();
        scores.push(score);
    }

    // Each failure should decrease the score.
    for window in scores.windows(2) {
        assert!(
            window[1] < window[0],
            "Score should decrease: {} -> {}",
            window[0],
            window[1]
        );
    }

    // After 20 failures: Beta(1000, 21), score = 1000/1021 = 0.97943...
    assert!(scores.last().unwrap() < &0.98);
}

#[test]
fn test_threshold_transitions_in_order() {
    let attributor = BayesianAttributor::new(default_config());

    // Record failures until we pass each threshold.
    let mut tiers_seen = Vec::new();
    for _ in 0..100 {
        let tier = attributor.record_probe_failure("gpu-1", None);
        if tiers_seen.last() != Some(&tier) {
            tiers_seen.push(tier);
        }
    }

    // We should see: Healthy -> IncreasedProbing -> Suspect -> Quarantine -> Condemned
    // (though some may be skipped if a single failure crosses multiple thresholds)
    assert!(!tiers_seen.is_empty());
    assert_eq!(*tiers_seen.first().unwrap(), ReliabilityTier::Healthy);

    // The last tier should be at least Quarantine.
    let last = *tiers_seen.last().unwrap();
    assert!(
        last == ReliabilityTier::Quarantine || last == ReliabilityTier::Condemned,
        "Expected Quarantine or Condemned, got {:?}",
        last
    );
}

#[test]
fn test_anomaly_weight_is_less_than_failure() {
    let attributor = BayesianAttributor::new(default_config());

    // 10 anomalies.
    for _ in 0..10 {
        attributor.record_anomaly("gpu-a");
    }

    // 10 failures.
    for _ in 0..10 {
        attributor.record_probe_failure("gpu-b", None);
    }

    let score_a = attributor.gpu_reliability_score("gpu-a").unwrap();
    let score_b = attributor.gpu_reliability_score("gpu-b").unwrap();

    // GPU with anomalies should have higher reliability (less degraded)
    // because anomaly weight (0.3) < failure weight (1.0).
    assert!(
        score_a > score_b,
        "Anomaly GPU should have higher score: {} vs {}",
        score_a,
        score_b
    );
}

#[test]
fn test_sm_level_isolation() {
    let attributor = BayesianAttributor::new(default_config());

    // Fail SM 5 multiple times.
    for _ in 0..10 {
        attributor.record_probe_failure("gpu-1", Some(5));
    }

    // SM 5 should be degraded.
    let sm5_belief = attributor.sm_belief("gpu-1", 5).unwrap();
    assert!(sm5_belief.reliability_score() < 0.99);

    // SM 6 should not be affected.
    assert!(attributor.sm_belief("gpu-1", 6).is_none());

    // GPU-level should also be degraded.
    let gpu_score = attributor.gpu_reliability_score("gpu-1").unwrap();
    assert!(gpu_score < 0.99);
}

#[test]
fn test_success_improves_reliability() {
    let attributor = BayesianAttributor::new(default_config());

    // Degrade with failures.
    for _ in 0..5 {
        attributor.record_probe_failure("gpu-1", None);
    }
    let degraded_score = attributor.gpu_reliability_score("gpu-1").unwrap();

    // Recover with successes.
    for _ in 0..100 {
        attributor.record_probe_success("gpu-1", None);
    }
    let recovered_score = attributor.gpu_reliability_score("gpu-1").unwrap();

    assert!(
        recovered_score > degraded_score,
        "Score should improve with successes: {} -> {}",
        degraded_score,
        recovered_score
    );
}

#[test]
fn test_variance_decreases_with_data() {
    let initial = BayesianBelief::new(1000.0, 1.0);
    let initial_var = initial.variance();

    let attributor = BayesianAttributor::new(default_config());
    for _ in 0..1000 {
        attributor.record_probe_success("gpu-1", None);
    }
    let updated = attributor.gpu_belief("gpu-1").unwrap();

    assert!(
        updated.variance() < initial_var,
        "Variance should decrease: {} -> {}",
        initial_var,
        updated.variance()
    );
}

#[test]
fn test_credible_interval() {
    let belief = BayesianBelief::new(1000.0, 1.0);
    let lower_95 = belief.lower_credible_bound(0.95).unwrap();
    let lower_99 = belief.lower_credible_bound(0.99).unwrap();

    // 95% lower bound should be higher (less conservative) than 99%.
    assert!(lower_95 > lower_99);
    // Both should be below the mean.
    assert!(lower_95 < belief.reliability_score());
    assert!(lower_99 < belief.reliability_score());
}

#[test]
fn test_reset_restores_prior() {
    let attributor = BayesianAttributor::new(default_config());

    for _ in 0..50 {
        attributor.record_probe_failure("gpu-1", None);
    }
    assert!(attributor.gpu_reliability_score("gpu-1").unwrap() < 0.97);

    attributor.reset_gpu_belief("gpu-1");

    let belief = attributor.gpu_belief("gpu-1").unwrap();
    assert_eq!(belief.alpha, 1000.0);
    assert_eq!(belief.beta, 1.0);
}

#[test]
fn test_tracked_gpus() {
    let attributor = BayesianAttributor::new(default_config());
    attributor.record_probe_success("gpu-1", None);
    attributor.record_probe_success("gpu-2", None);
    attributor.record_probe_success("gpu-3", None);

    let gpus = attributor.tracked_gpus();
    assert_eq!(gpus.len(), 3);
}
