//! Bayesian attribution model for GPU reliability estimation.
//!
//! Each GPU (and optionally each SM within a GPU) maintains a Beta distribution
//! representing the posterior probability that the unit is reliable. The model
//! uses conjugate Bayesian updating:
//!
//! ```text
//! P(reliable | evidence) ~ Beta(alpha, beta)
//!   alpha = prior_alpha + sum(successful_probes)
//!   beta  = prior_beta  + sum(weighted_failures)
//! ```
//!
//! Probe failures contribute weight 1.0 to beta; anomaly events contribute
//! a configurable weight (default 0.3) reflecting lower individual confidence.
//!
//! The reliability score is the mean of the Beta distribution: alpha / (alpha + beta).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Beta, ContinuousCDF};
use thiserror::Error;

use crate::util::config::BayesianConfig;

/// Errors that can occur in Bayesian attribution operations.
#[derive(Debug, Error)]
pub enum BayesianError {
    #[error("invalid Beta distribution parameters: alpha={alpha}, beta={beta}")]
    InvalidParameters { alpha: f64, beta: f64 },

    #[error("GPU {gpu_id} not found in attribution model")]
    GpuNotFound { gpu_id: String },
}

/// The health tier derived from the Bayesian reliability score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ReliabilityTier {
    /// Fully healthy, no concerns.
    Healthy,
    /// Reliability dipped below the increased-probing threshold.
    IncreasedProbing,
    /// Reliability dipped below the suspect threshold; TMR validation needed.
    Suspect,
    /// Reliability dipped below the quarantine threshold; remove from production.
    Quarantine,
    /// Reliability dipped below the condemned threshold; permanent removal.
    Condemned,
}

impl std::fmt::Display for ReliabilityTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReliabilityTier::Healthy => write!(f, "HEALTHY"),
            ReliabilityTier::IncreasedProbing => write!(f, "INCREASED_PROBING"),
            ReliabilityTier::Suspect => write!(f, "SUSPECT"),
            ReliabilityTier::Quarantine => write!(f, "QUARANTINE"),
            ReliabilityTier::Condemned => write!(f, "CONDEMNED"),
        }
    }
}

/// Bayesian belief state for a single unit (GPU or SM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianBelief {
    /// Current alpha parameter of the Beta distribution.
    pub alpha: f64,

    /// Current beta parameter of the Beta distribution.
    pub beta: f64,

    /// Total number of successful probes recorded.
    pub total_successes: u64,

    /// Total number of probe failures recorded.
    pub total_failures: u64,

    /// Total number of anomaly events recorded.
    pub total_anomalies: u64,

    /// Timestamp of the last update to this belief.
    pub last_updated: DateTime<Utc>,
}

impl BayesianBelief {
    /// Create a new belief with the given prior parameters.
    pub fn new(prior_alpha: f64, prior_beta: f64) -> Self {
        Self {
            alpha: prior_alpha,
            beta: prior_beta,
            total_successes: 0,
            total_failures: 0,
            total_anomalies: 0,
            last_updated: Utc::now(),
        }
    }

    /// Compute the reliability score: mean of the Beta distribution.
    ///
    /// Returns alpha / (alpha + beta), which is the expected probability
    /// that the next probe will succeed.
    pub fn reliability_score(&self) -> f64 {
        if self.alpha + self.beta == 0.0 {
            return 0.5; // Degenerate case; should never happen with positive priors.
        }
        self.alpha / (self.alpha + self.beta)
    }

    /// Compute the lower bound of a credible interval at the given confidence level.
    ///
    /// For example, `lower_credible_bound(0.95)` returns the value x such that
    /// P(reliability > x) = 0.95 according to the Beta posterior.
    pub fn lower_credible_bound(&self, confidence: f64) -> Result<f64, BayesianError> {
        let dist = Beta::new(self.alpha, self.beta).map_err(|_| {
            BayesianError::InvalidParameters {
                alpha: self.alpha,
                beta: self.beta,
            }
        })?;
        Ok(dist.inverse_cdf(1.0 - confidence))
    }

    /// Compute the variance of the Beta distribution.
    ///
    /// Lower variance means higher confidence in the estimate.
    pub fn variance(&self) -> f64 {
        let sum = self.alpha + self.beta;
        if sum == 0.0 || sum + 1.0 == 0.0 {
            return f64::MAX;
        }
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }
}

/// The Bayesian attribution engine that tracks beliefs for all GPUs and SMs.
#[derive(Debug)]
pub struct BayesianAttributor {
    /// Configuration parameters.
    config: BayesianConfig,

    /// Per-GPU beliefs, keyed by GPU UUID.
    gpu_beliefs: dashmap::DashMap<String, BayesianBelief>,

    /// Per-SM beliefs, keyed by "gpu_uuid:sm_id".
    sm_beliefs: dashmap::DashMap<String, BayesianBelief>,
}

impl BayesianAttributor {
    /// Create a new Bayesian attributor with the given configuration.
    pub fn new(config: BayesianConfig) -> Self {
        Self {
            config,
            gpu_beliefs: dashmap::DashMap::new(),
            sm_beliefs: dashmap::DashMap::new(),
        }
    }

    /// Get or create the belief state for a GPU.
    pub fn get_or_create_gpu_belief(&self, gpu_uuid: &str) -> BayesianBelief {
        self.gpu_beliefs
            .entry(gpu_uuid.to_string())
            .or_insert_with(|| {
                BayesianBelief::new(self.config.prior_alpha, self.config.prior_beta)
            })
            .clone()
    }

    /// Get or create the belief state for a specific SM.
    fn get_or_create_sm_belief(&self, gpu_uuid: &str, sm_id: u32) -> BayesianBelief {
        let key = format!("{}:{}", gpu_uuid, sm_id);
        self.sm_beliefs
            .entry(key)
            .or_insert_with(|| {
                BayesianBelief::new(self.config.prior_alpha, self.config.prior_beta)
            })
            .clone()
    }

    /// Record a successful probe execution for a GPU (and optionally an SM).
    ///
    /// This increments the alpha (success) parameter by 1.0.
    pub fn record_probe_success(&self, gpu_uuid: &str, sm_id: Option<u32>) -> ReliabilityTier {
        // Update GPU-level belief.
        let mut gpu_entry = self
            .gpu_beliefs
            .entry(gpu_uuid.to_string())
            .or_insert_with(|| {
                BayesianBelief::new(self.config.prior_alpha, self.config.prior_beta)
            });
        gpu_entry.alpha += 1.0;
        gpu_entry.total_successes += 1;
        gpu_entry.last_updated = Utc::now();
        let score = gpu_entry.reliability_score();
        drop(gpu_entry);

        // Update SM-level belief if SM is specified.
        if let Some(sm) = sm_id {
            let key = format!("{}:{}", gpu_uuid, sm);
            let mut sm_entry = self.sm_beliefs.entry(key).or_insert_with(|| {
                BayesianBelief::new(self.config.prior_alpha, self.config.prior_beta)
            });
            sm_entry.alpha += 1.0;
            sm_entry.total_successes += 1;
            sm_entry.last_updated = Utc::now();
        }

        self.score_to_tier(score)
    }

    /// Record a probe failure for a GPU (and optionally an SM).
    ///
    /// This increments the beta (failure) parameter by the configured probe failure weight.
    pub fn record_probe_failure(&self, gpu_uuid: &str, sm_id: Option<u32>) -> ReliabilityTier {
        let weight = self.config.probe_failure_weight;

        let mut gpu_entry = self
            .gpu_beliefs
            .entry(gpu_uuid.to_string())
            .or_insert_with(|| {
                BayesianBelief::new(self.config.prior_alpha, self.config.prior_beta)
            });
        gpu_entry.beta += weight;
        gpu_entry.total_failures += 1;
        gpu_entry.last_updated = Utc::now();
        let score = gpu_entry.reliability_score();
        drop(gpu_entry);

        if let Some(sm) = sm_id {
            let key = format!("{}:{}", gpu_uuid, sm);
            let mut sm_entry = self.sm_beliefs.entry(key).or_insert_with(|| {
                BayesianBelief::new(self.config.prior_alpha, self.config.prior_beta)
            });
            sm_entry.beta += weight;
            sm_entry.total_failures += 1;
            sm_entry.last_updated = Utc::now();
        }

        self.score_to_tier(score)
    }

    /// Record an anomaly event attributed to a GPU.
    ///
    /// Anomaly events contribute the configured anomaly weight to beta,
    /// reflecting lower individual confidence than a direct probe failure.
    pub fn record_anomaly(&self, gpu_uuid: &str) -> ReliabilityTier {
        let weight = self.config.anomaly_weight;

        let mut gpu_entry = self
            .gpu_beliefs
            .entry(gpu_uuid.to_string())
            .or_insert_with(|| {
                BayesianBelief::new(self.config.prior_alpha, self.config.prior_beta)
            });
        gpu_entry.beta += weight;
        gpu_entry.total_anomalies += 1;
        gpu_entry.last_updated = Utc::now();
        let score = gpu_entry.reliability_score();
        drop(gpu_entry);

        self.score_to_tier(score)
    }

    /// Determine the reliability tier from a reliability score.
    pub fn score_to_tier(&self, score: f64) -> ReliabilityTier {
        if score < self.config.condemned_threshold {
            ReliabilityTier::Condemned
        } else if score < self.config.quarantine_threshold {
            ReliabilityTier::Quarantine
        } else if score < self.config.suspect_threshold {
            ReliabilityTier::Suspect
        } else if score < self.config.increased_probing_threshold {
            ReliabilityTier::IncreasedProbing
        } else {
            ReliabilityTier::Healthy
        }
    }

    /// Get the current reliability score for a GPU. Returns `None` if no data exists.
    pub fn gpu_reliability_score(&self, gpu_uuid: &str) -> Option<f64> {
        self.gpu_beliefs
            .get(gpu_uuid)
            .map(|b| b.reliability_score())
    }

    /// Get the current belief state for a GPU. Returns `None` if no data exists.
    pub fn gpu_belief(&self, gpu_uuid: &str) -> Option<BayesianBelief> {
        self.gpu_beliefs.get(gpu_uuid).map(|b| b.clone())
    }

    /// Get the current belief state for an SM. Returns `None` if no data exists.
    pub fn sm_belief(&self, gpu_uuid: &str, sm_id: u32) -> Option<BayesianBelief> {
        let key = format!("{}:{}", gpu_uuid, sm_id);
        self.sm_beliefs.get(&key).map(|b| b.clone())
    }

    /// Get the current reliability tier for a GPU.
    pub fn gpu_tier(&self, gpu_uuid: &str) -> ReliabilityTier {
        match self.gpu_reliability_score(gpu_uuid) {
            Some(score) => self.score_to_tier(score),
            None => ReliabilityTier::Healthy, // No data yet means assumed healthy.
        }
    }

    /// List all tracked GPU UUIDs.
    pub fn tracked_gpus(&self) -> Vec<String> {
        self.gpu_beliefs
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Restore a GPU's belief state (e.g., from Redis on startup).
    pub fn restore_gpu_belief(&self, gpu_uuid: &str, belief: BayesianBelief) {
        self.gpu_beliefs.insert(gpu_uuid.to_string(), belief);
    }

    /// Restore an SM's belief state.
    pub fn restore_sm_belief(&self, gpu_uuid: &str, sm_id: u32, belief: BayesianBelief) {
        let key = format!("{}:{}", gpu_uuid, sm_id);
        self.sm_beliefs.insert(key, belief);
    }

    /// Reset a GPU's belief to the prior (used after reinstatement with fresh state).
    pub fn reset_gpu_belief(&self, gpu_uuid: &str) {
        self.gpu_beliefs.insert(
            gpu_uuid.to_string(),
            BayesianBelief::new(self.config.prior_alpha, self.config.prior_beta),
        );
    }

    /// Get the number of tracked GPUs.
    pub fn gpu_count(&self) -> usize {
        self.gpu_beliefs.len()
    }

    /// Get the configuration.
    pub fn config(&self) -> &BayesianConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BayesianConfig {
        BayesianConfig::default()
    }

    #[test]
    fn test_initial_belief_is_healthy() {
        let attributor = BayesianAttributor::new(test_config());
        let belief = attributor.get_or_create_gpu_belief("gpu-1");
        assert!((belief.reliability_score() - (1000.0 / 1001.0)).abs() < 1e-10);
        assert_eq!(attributor.gpu_tier("gpu-1"), ReliabilityTier::Healthy);
    }

    #[test]
    fn test_probe_success_increases_alpha() {
        let attributor = BayesianAttributor::new(test_config());
        attributor.record_probe_success("gpu-1", None);
        let belief = attributor.gpu_belief("gpu-1").unwrap();
        assert_eq!(belief.alpha, 1001.0);
        assert_eq!(belief.beta, 1.0);
        assert_eq!(belief.total_successes, 1);
    }

    #[test]
    fn test_probe_failure_increases_beta() {
        let attributor = BayesianAttributor::new(test_config());
        attributor.record_probe_failure("gpu-1", None);
        let belief = attributor.gpu_belief("gpu-1").unwrap();
        assert_eq!(belief.alpha, 1000.0);
        assert_eq!(belief.beta, 2.0);
        assert_eq!(belief.total_failures, 1);
    }

    #[test]
    fn test_anomaly_adds_fractional_weight() {
        let attributor = BayesianAttributor::new(test_config());
        attributor.record_anomaly("gpu-1");
        let belief = attributor.gpu_belief("gpu-1").unwrap();
        assert!((belief.beta - 1.3).abs() < 1e-10); // 1.0 prior + 0.3 anomaly
    }

    #[test]
    fn test_many_failures_cause_quarantine() {
        let attributor = BayesianAttributor::new(test_config());
        // With prior Beta(1000, 1), we need enough failures to push below 0.99.
        // reliability = 1000 / (1000 + 1 + n) < 0.99
        // => 1000 < 0.99 * (1001 + n)
        // => 1000 / 0.99 < 1001 + n
        // => 1010.10... < 1001 + n
        // => n > 9.1
        // So 10 failures should push below 0.99.
        for _ in 0..11 {
            attributor.record_probe_failure("gpu-1", None);
        }
        let tier = attributor.gpu_tier("gpu-1");
        assert!(
            tier == ReliabilityTier::Quarantine || tier == ReliabilityTier::Condemned,
            "Expected Quarantine or Condemned, got {:?}",
            tier
        );
    }

    #[test]
    fn test_sm_level_tracking() {
        let attributor = BayesianAttributor::new(test_config());
        attributor.record_probe_failure("gpu-1", Some(5));
        let sm_belief = attributor.sm_belief("gpu-1", 5).unwrap();
        assert_eq!(sm_belief.total_failures, 1);
        assert_eq!(sm_belief.beta, 2.0);

        // SM 6 should still be at prior.
        assert!(attributor.sm_belief("gpu-1", 6).is_none());
    }

    #[test]
    fn test_variance_decreases_with_evidence() {
        let attributor = BayesianAttributor::new(test_config());
        let initial = attributor.get_or_create_gpu_belief("gpu-1");
        let initial_var = initial.variance();

        for _ in 0..100 {
            attributor.record_probe_success("gpu-1", None);
        }
        let updated = attributor.gpu_belief("gpu-1").unwrap();
        assert!(updated.variance() < initial_var);
    }

    #[test]
    fn test_lower_credible_bound() {
        let belief = BayesianBelief::new(1000.0, 1.0);
        let lb = belief.lower_credible_bound(0.95).unwrap();
        assert!(lb > 0.99);
        assert!(lb < belief.reliability_score());
    }

    #[test]
    fn test_reset_belief() {
        let attributor = BayesianAttributor::new(test_config());
        for _ in 0..100 {
            attributor.record_probe_failure("gpu-1", None);
        }
        attributor.reset_gpu_belief("gpu-1");
        let belief = attributor.gpu_belief("gpu-1").unwrap();
        assert_eq!(belief.alpha, 1000.0);
        assert_eq!(belief.beta, 1.0);
    }
}
