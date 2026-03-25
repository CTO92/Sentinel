//! Triple Modular Redundancy (TMR) scheduler.
//!
//! Periodically selects triples of GPUs for canary computations where the
//! same input is run on three GPUs and outputs compared. The scheduler
//! optimizes for trust graph coverage while prioritizing suspect GPUs.
//!
//! The scheduler runs as a periodic Tokio task, emitting `TmrCanaryRequest`
//! messages to be dispatched to the probe agents.

use std::sync::Arc;

use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::health::quarantine::{QuarantineManager, QuarantineState};
use crate::trust::trust_graph::TrustGraph;
use crate::util::config::TmrConfig;

/// A request to perform a TMR canary computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmrCanaryRequest {
    /// Unique identifier for this canary run.
    pub canary_id: String,

    /// The three GPUs to run the computation on.
    pub gpu_uuids: [String; 3],

    /// Timeout for each GPU's computation in milliseconds.
    pub timeout_ms: u32,
}

/// Result of a TMR canary run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmrCanaryResult {
    /// Canary run identifier.
    pub canary_id: String,

    /// Per-GPU output fingerprints (SHA-256).
    pub fingerprints: Vec<(String, Vec<u8>)>,

    /// The consensus fingerprint (majority vote).
    pub consensus: Option<Vec<u8>>,

    /// The GPU that dissented from the majority, if any.
    pub dissenting_gpu: Option<String>,

    /// Whether all three GPUs agreed.
    pub unanimous: bool,
}

/// The TMR scheduler responsible for selecting and scheduling canary runs.
pub struct TmrScheduler {
    /// Configuration.
    config: TmrConfig,

    /// Reference to the trust graph for partner selection.
    trust_graph: Arc<RwLock<TrustGraph>>,

    /// Reference to the quarantine manager for GPU state awareness.
    quarantine_manager: Arc<RwLock<QuarantineManager>>,
}

impl TmrScheduler {
    /// Create a new TMR scheduler.
    pub fn new(
        config: TmrConfig,
        trust_graph: Arc<RwLock<TrustGraph>>,
        quarantine_manager: Arc<RwLock<QuarantineManager>>,
    ) -> Self {
        Self {
            config,
            trust_graph,
            quarantine_manager,
        }
    }

    /// Generate the next batch of TMR canary requests.
    ///
    /// Prioritizes:
    /// 1. GPUs in SUSPECT state (need TMR validation)
    /// 2. GPU pairs with low coverage in the trust graph
    /// 3. Random pairs for baseline coverage expansion
    pub async fn schedule_canaries(&self) -> Vec<TmrCanaryRequest> {
        let trust = self.trust_graph.read().await;
        let qm = self.quarantine_manager.read().await;

        let all_gpus = trust.all_gpus();
        if all_gpus.len() < 3 {
            debug!("Not enough GPUs for TMR (need at least 3, have {})", all_gpus.len());
            return Vec::new();
        }

        // Partition GPUs by state.
        let healthy_gpus: Vec<String> = all_gpus
            .iter()
            .filter(|g| qm.get_state(g) == QuarantineState::Healthy)
            .cloned()
            .collect();

        let suspect_gpus: Vec<String> = all_gpus
            .iter()
            .filter(|g| qm.get_state(g) == QuarantineState::Suspect)
            .cloned()
            .collect();

        let mut requests = Vec::new();
        let mut remaining = self.config.triples_per_interval;

        // Priority 1: Schedule TMR for suspect GPUs.
        for suspect_gpu in &suspect_gpus {
            if remaining == 0 {
                break;
            }

            if let Some((partner_a, partner_b)) =
                trust.select_tmr_partners(suspect_gpu, &healthy_gpus)
            {
                requests.push(TmrCanaryRequest {
                    canary_id: uuid::Uuid::new_v4().to_string(),
                    gpu_uuids: [suspect_gpu.clone(), partner_a, partner_b],
                    timeout_ms: self.config.timeout_ms,
                });
                remaining -= 1;
            }
        }

        // Priority 2: Coverage expansion - select GPUs with least coverage.
        let mut rng = rand::thread_rng();
        let mut coverage_candidates = healthy_gpus.clone();
        coverage_candidates.shuffle(&mut rng);

        for target in coverage_candidates.iter().take(remaining) {
            if let Some((partner_a, partner_b)) =
                trust.select_tmr_partners(target, &healthy_gpus)
            {
                requests.push(TmrCanaryRequest {
                    canary_id: uuid::Uuid::new_v4().to_string(),
                    gpu_uuids: [target.clone(), partner_a, partner_b],
                    timeout_ms: self.config.timeout_ms,
                });
            }
        }

        if !requests.is_empty() {
            info!(
                count = requests.len(),
                suspect_count = suspect_gpus.len(),
                "Scheduled TMR canary runs"
            );
        }

        requests
    }

    /// Process the result of a TMR canary run, updating the trust graph.
    pub async fn process_result(&self, result: TmrCanaryResult) {
        let mut trust = self.trust_graph.write().await;

        if result.unanimous {
            // All three agreed: record pairwise agreements.
            let gpus = &result.fingerprints;
            if gpus.len() == 3 {
                trust.record_agreement(&gpus[0].0, &gpus[1].0);
                trust.record_agreement(&gpus[0].0, &gpus[2].0);
                trust.record_agreement(&gpus[1].0, &gpus[2].0);

                debug!(
                    canary_id = %result.canary_id,
                    "TMR unanimous agreement"
                );
            }
        } else if let Some(ref dissenter) = result.dissenting_gpu {
            // One GPU dissented: record disagreements with the dissenter
            // and agreement between the two that agreed.
            let agreeing: Vec<&(String, Vec<u8>)> = result
                .fingerprints
                .iter()
                .filter(|(gpu, _)| gpu != dissenter)
                .collect();

            if agreeing.len() == 2 {
                trust.record_agreement(&agreeing[0].0, &agreeing[1].0);
                trust.record_disagreement(dissenter, &agreeing[0].0);
                trust.record_disagreement(dissenter, &agreeing[1].0);

                warn!(
                    canary_id = %result.canary_id,
                    dissenter = %dissenter,
                    "TMR dissent detected"
                );
            }
        }

        // Update coverage metric.
        let coverage = trust.coverage_percent();
        crate::util::metrics::set_trust_graph_coverage(coverage);
    }

    /// Get the scheduling interval.
    pub fn interval_secs(&self) -> u64 {
        self.config.interval_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::config::{QuarantineConfig, TrustConfig};

    #[tokio::test]
    async fn test_schedule_with_few_gpus() {
        let trust = Arc::new(RwLock::new(TrustGraph::new(TrustConfig::default())));
        let qm = Arc::new(RwLock::new(QuarantineManager::new(
            QuarantineConfig::default(),
        )));

        let scheduler = TmrScheduler::new(TmrConfig::default(), trust, qm);
        let requests = scheduler.schedule_canaries().await;
        // Not enough GPUs, should return empty.
        assert!(requests.is_empty());
    }

    #[tokio::test]
    async fn test_schedule_with_enough_gpus() {
        let trust = Arc::new(RwLock::new(TrustGraph::new(TrustConfig {
            min_comparisons: 1,
            decay_factor: 0.999,
        })));
        let qm = Arc::new(RwLock::new(QuarantineManager::new(
            QuarantineConfig::default(),
        )));

        // Register GPUs.
        {
            let mut t = trust.write().await;
            for i in 1..=5 {
                t.register_gpu(&format!("gpu-{}", i));
            }
        }

        let scheduler = TmrScheduler::new(
            TmrConfig {
                interval_secs: 60,
                triples_per_interval: 3,
                timeout_ms: 5000,
            },
            trust,
            qm,
        );
        let requests = scheduler.schedule_canaries().await;
        assert!(!requests.is_empty());
        assert!(requests.len() <= 3);
    }

    #[tokio::test]
    async fn test_process_unanimous_result() {
        let trust = Arc::new(RwLock::new(TrustGraph::new(TrustConfig {
            min_comparisons: 1,
            decay_factor: 0.999,
        })));
        let qm = Arc::new(RwLock::new(QuarantineManager::new(
            QuarantineConfig::default(),
        )));

        let scheduler = TmrScheduler::new(TmrConfig::default(), trust.clone(), qm);

        let result = TmrCanaryResult {
            canary_id: "test".to_string(),
            fingerprints: vec![
                ("gpu-1".to_string(), vec![1, 2, 3]),
                ("gpu-2".to_string(), vec![1, 2, 3]),
                ("gpu-3".to_string(), vec![1, 2, 3]),
            ],
            consensus: Some(vec![1, 2, 3]),
            dissenting_gpu: None,
            unanimous: true,
        };

        scheduler.process_result(result).await;

        let t = trust.read().await;
        assert_eq!(t.trust_score("gpu-1", "gpu-2"), Some(1.0));
        assert_eq!(t.trust_score("gpu-1", "gpu-3"), Some(1.0));
    }

    #[tokio::test]
    async fn test_process_dissent_result() {
        let trust = Arc::new(RwLock::new(TrustGraph::new(TrustConfig {
            min_comparisons: 1,
            decay_factor: 0.999,
        })));
        let qm = Arc::new(RwLock::new(QuarantineManager::new(
            QuarantineConfig::default(),
        )));

        let scheduler = TmrScheduler::new(TmrConfig::default(), trust.clone(), qm);

        let result = TmrCanaryResult {
            canary_id: "test".to_string(),
            fingerprints: vec![
                ("gpu-1".to_string(), vec![1, 2, 3]),
                ("gpu-2".to_string(), vec![1, 2, 3]),
                ("gpu-3".to_string(), vec![4, 5, 6]),
            ],
            consensus: Some(vec![1, 2, 3]),
            dissenting_gpu: Some("gpu-3".to_string()),
            unanimous: false,
        };

        scheduler.process_result(result).await;

        let t = trust.read().await;
        assert_eq!(t.trust_score("gpu-1", "gpu-2"), Some(1.0));
        // Dissenter should have 0.0 trust with the agreeing GPUs.
        assert_eq!(t.trust_score("gpu-1", "gpu-3"), Some(0.0));
    }
}
