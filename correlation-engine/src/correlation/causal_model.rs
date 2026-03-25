//! Causal model for environmental factor attribution.
//!
//! This module implements a Directed Acyclic Graph (DAG) linking environmental
//! factors to SDC probability. Factors include temperature, voltage, uptime,
//! firmware version, and manufacturing lot. The model updates causal weights
//! based on observed correlations between environmental readings and SDC events.
//!
//! The causal model helps distinguish between:
//! - Environmental transients (thermal, power) that may resolve on their own
//! - Persistent hardware defects that require quarantine
//! - Firmware/software bugs that affect specific versions

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Environmental factors that can influence SDC probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalFactor {
    /// GPU die temperature.
    Temperature,
    /// GPU core voltage.
    Voltage,
    /// GPU uptime since last reset.
    Uptime,
    /// GPU firmware/VBIOS version.
    FirmwareVersion,
    /// GPU manufacturing lot/batch.
    ManufacturingLot,
    /// Power delivery anomaly.
    PowerDelivery,
    /// Memory (HBM) temperature.
    MemoryTemperature,
    /// ECC error accumulation.
    EccErrors,
    /// PCIe link degradation.
    PcieLinkDegradation,
    /// NVLink errors.
    NvLinkErrors,
}

impl std::fmt::Display for CausalFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CausalFactor::Temperature => write!(f, "temperature"),
            CausalFactor::Voltage => write!(f, "voltage"),
            CausalFactor::Uptime => write!(f, "uptime"),
            CausalFactor::FirmwareVersion => write!(f, "firmware_version"),
            CausalFactor::ManufacturingLot => write!(f, "manufacturing_lot"),
            CausalFactor::PowerDelivery => write!(f, "power_delivery"),
            CausalFactor::MemoryTemperature => write!(f, "memory_temperature"),
            CausalFactor::EccErrors => write!(f, "ecc_errors"),
            CausalFactor::PcieLinkDegradation => write!(f, "pcie_link_degradation"),
            CausalFactor::NvLinkErrors => write!(f, "nvlink_errors"),
        }
    }
}

/// An observation of an environmental factor at the time of an SDC event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorObservation {
    /// The factor observed.
    pub factor: CausalFactor,

    /// The value of the factor (normalized to [0, 1] range for comparison).
    pub normalized_value: f64,

    /// Whether this observation was during an SDC event (true) or a normal
    /// period (false).
    pub during_sdc: bool,

    /// Timestamp of the observation.
    pub timestamp: DateTime<Utc>,

    /// GPU UUID associated with this observation.
    pub gpu_uuid: String,
}

/// A causal edge in the DAG, representing the relationship between a factor
/// and SDC probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    /// The causal factor.
    pub factor: CausalFactor,

    /// Causal weight: how strongly this factor correlates with SDC.
    /// Range [0.0, 1.0]; higher means stronger causal link.
    pub weight: f64,

    /// Number of SDC events where this factor was elevated.
    pub sdc_elevated_count: u64,

    /// Number of normal periods where this factor was elevated.
    pub normal_elevated_count: u64,

    /// Total SDC events observed.
    pub total_sdc_events: u64,

    /// Total normal observations.
    pub total_normal_observations: u64,

    /// Last time this edge was updated.
    pub last_updated: DateTime<Utc>,
}

impl CausalEdge {
    /// Create a new causal edge with zero observations.
    fn new(factor: CausalFactor) -> Self {
        Self {
            factor,
            weight: 0.0,
            sdc_elevated_count: 0,
            normal_elevated_count: 0,
            total_sdc_events: 0,
            total_normal_observations: 0,
            last_updated: Utc::now(),
        }
    }

    /// Recalculate the causal weight using a simple risk-ratio approach.
    ///
    /// weight = P(factor_elevated | SDC) / P(factor_elevated | normal)
    /// Normalized to [0, 1] using a sigmoid-like transform.
    fn recalculate_weight(&mut self) {
        if self.total_sdc_events == 0 || self.total_normal_observations == 0 {
            self.weight = 0.0;
            return;
        }

        let p_elevated_given_sdc =
            self.sdc_elevated_count as f64 / self.total_sdc_events as f64;
        let p_elevated_given_normal =
            self.normal_elevated_count as f64 / self.total_normal_observations as f64;

        // Avoid division by zero: if factor is never elevated during normal ops,
        // the risk ratio is infinite; cap at a high value.
        let risk_ratio = if p_elevated_given_normal < 1e-10 {
            if self.sdc_elevated_count > 0 {
                100.0
            } else {
                0.0
            }
        } else {
            p_elevated_given_sdc / p_elevated_given_normal
        };

        // Sigmoid normalization: weight = 1 - 1/(1 + risk_ratio/10)
        // This maps risk_ratio of 0 -> 0, 10 -> 0.5, infinity -> 1.0
        self.weight = 1.0 - 1.0 / (1.0 + risk_ratio / 10.0);
        self.last_updated = Utc::now();
    }
}

/// The causal model DAG for a GPU.
///
/// Each GPU can have different causal weights because the environmental
/// conditions (cooling, PSU) vary by physical location.
#[derive(Debug)]
pub struct CausalModel {
    /// Per-GPU causal edges. Outer key: GPU UUID, inner key: CausalFactor.
    edges: HashMap<String, HashMap<CausalFactor, CausalEdge>>,

    /// Fleet-wide causal edges (aggregated across all GPUs).
    fleet_edges: HashMap<CausalFactor, CausalEdge>,

    /// Threshold above which a factor's normalized value is considered "elevated".
    elevation_threshold: f64,
}

impl CausalModel {
    /// Create a new causal model.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            fleet_edges: HashMap::new(),
            elevation_threshold: 0.7,
        }
    }

    /// Create a new causal model with a custom elevation threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            edges: HashMap::new(),
            fleet_edges: HashMap::new(),
            elevation_threshold: threshold,
        }
    }

    /// Record a factor observation and update causal weights.
    pub fn record_observation(&mut self, obs: FactorObservation) {
        let is_elevated = obs.normalized_value >= self.elevation_threshold;

        // Update per-GPU edge.
        let gpu_edges = self.edges.entry(obs.gpu_uuid.clone()).or_default();
        let edge = gpu_edges
            .entry(obs.factor)
            .or_insert_with(|| CausalEdge::new(obs.factor));

        if obs.during_sdc {
            edge.total_sdc_events += 1;
            if is_elevated {
                edge.sdc_elevated_count += 1;
            }
        } else {
            edge.total_normal_observations += 1;
            if is_elevated {
                edge.normal_elevated_count += 1;
            }
        }
        edge.recalculate_weight();

        // Update fleet-wide edge.
        let fleet_edge = self
            .fleet_edges
            .entry(obs.factor)
            .or_insert_with(|| CausalEdge::new(obs.factor));

        if obs.during_sdc {
            fleet_edge.total_sdc_events += 1;
            if is_elevated {
                fleet_edge.sdc_elevated_count += 1;
            }
        } else {
            fleet_edge.total_normal_observations += 1;
            if is_elevated {
                fleet_edge.normal_elevated_count += 1;
            }
        }
        fleet_edge.recalculate_weight();
    }

    /// Get the causal weight of a factor for a specific GPU.
    /// Falls back to fleet-wide weight if no GPU-specific data exists.
    pub fn factor_weight(&self, gpu_uuid: &str, factor: CausalFactor) -> f64 {
        self.edges
            .get(gpu_uuid)
            .and_then(|edges| edges.get(&factor))
            .map(|e| e.weight)
            .unwrap_or_else(|| {
                self.fleet_edges
                    .get(&factor)
                    .map(|e| e.weight)
                    .unwrap_or(0.0)
            })
    }

    /// Get all causal weights for a GPU, sorted by weight descending.
    pub fn gpu_factors(&self, gpu_uuid: &str) -> Vec<(CausalFactor, f64)> {
        let mut factors: Vec<(CausalFactor, f64)> = self
            .edges
            .get(gpu_uuid)
            .map(|edges| edges.iter().map(|(f, e)| (*f, e.weight)).collect())
            .unwrap_or_default();

        factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        factors
    }

    /// Get fleet-wide causal weights, sorted by weight descending.
    pub fn fleet_factors(&self) -> Vec<(CausalFactor, f64)> {
        let mut factors: Vec<(CausalFactor, f64)> = self
            .fleet_edges
            .iter()
            .map(|(f, e)| (*f, e.weight))
            .collect();

        factors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        factors
    }

    /// Compute the overall environmental risk score for a GPU given current
    /// factor values. This is a weighted sum of factor contributions.
    pub fn environmental_risk(
        &self,
        gpu_uuid: &str,
        current_factors: &HashMap<CausalFactor, f64>,
    ) -> f64 {
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for (factor, value) in current_factors {
            let w = self.factor_weight(gpu_uuid, *factor);
            total_weight += w;
            weighted_sum += w * value;
        }

        if total_weight == 0.0 {
            return 0.0;
        }

        weighted_sum / total_weight
    }

    /// Determine whether an SDC event is likely environmental (transient) or
    /// hardware-related (persistent).
    ///
    /// Returns the likelihood that the event is environmentally caused (0.0 to 1.0).
    pub fn environmental_likelihood(
        &self,
        gpu_uuid: &str,
        current_factors: &HashMap<CausalFactor, f64>,
    ) -> f64 {
        let environmental_factors = [
            CausalFactor::Temperature,
            CausalFactor::Voltage,
            CausalFactor::PowerDelivery,
            CausalFactor::MemoryTemperature,
        ];

        let mut env_score = 0.0;
        let mut env_count = 0;

        for factor in &environmental_factors {
            if let Some(value) = current_factors.get(factor) {
                let w = self.factor_weight(gpu_uuid, *factor);
                if *value >= self.elevation_threshold {
                    env_score += w;
                }
                env_count += 1;
            }
        }

        if env_count == 0 {
            return 0.0;
        }

        // Normalize to [0, 1].
        (env_score / env_count as f64).min(1.0)
    }

    /// Get the number of tracked GPUs.
    pub fn gpu_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for CausalModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_model_has_zero_weights() {
        let model = CausalModel::new();
        assert_eq!(
            model.factor_weight("gpu-1", CausalFactor::Temperature),
            0.0
        );
    }

    #[test]
    fn test_elevated_factor_during_sdc_increases_weight() {
        let mut model = CausalModel::new();

        // Record many normal observations with low temperature.
        for _ in 0..100 {
            model.record_observation(FactorObservation {
                factor: CausalFactor::Temperature,
                normalized_value: 0.3,
                during_sdc: false,
                timestamp: Utc::now(),
                gpu_uuid: "gpu-1".to_string(),
            });
        }

        // Record SDC events with high temperature.
        for _ in 0..10 {
            model.record_observation(FactorObservation {
                factor: CausalFactor::Temperature,
                normalized_value: 0.9,
                during_sdc: true,
                timestamp: Utc::now(),
                gpu_uuid: "gpu-1".to_string(),
            });
        }

        let weight = model.factor_weight("gpu-1", CausalFactor::Temperature);
        assert!(weight > 0.5, "Expected high weight, got {}", weight);
    }

    #[test]
    fn test_non_elevated_factor_during_sdc_stays_low() {
        let mut model = CausalModel::new();

        // Normal observations.
        for _ in 0..100 {
            model.record_observation(FactorObservation {
                factor: CausalFactor::Temperature,
                normalized_value: 0.3,
                during_sdc: false,
                timestamp: Utc::now(),
                gpu_uuid: "gpu-1".to_string(),
            });
        }

        // SDC events with normal temperature.
        for _ in 0..10 {
            model.record_observation(FactorObservation {
                factor: CausalFactor::Temperature,
                normalized_value: 0.3,
                during_sdc: true,
                timestamp: Utc::now(),
                gpu_uuid: "gpu-1".to_string(),
            });
        }

        let weight = model.factor_weight("gpu-1", CausalFactor::Temperature);
        assert!(weight < 0.3, "Expected low weight, got {}", weight);
    }

    #[test]
    fn test_environmental_likelihood() {
        let mut model = CausalModel::new();

        // Build up temperature as a strong causal factor.
        for _ in 0..100 {
            model.record_observation(FactorObservation {
                factor: CausalFactor::Temperature,
                normalized_value: 0.3,
                during_sdc: false,
                timestamp: Utc::now(),
                gpu_uuid: "gpu-1".to_string(),
            });
        }
        for _ in 0..50 {
            model.record_observation(FactorObservation {
                factor: CausalFactor::Temperature,
                normalized_value: 0.9,
                during_sdc: true,
                timestamp: Utc::now(),
                gpu_uuid: "gpu-1".to_string(),
            });
        }

        let mut current = HashMap::new();
        current.insert(CausalFactor::Temperature, 0.9);

        let likelihood = model.environmental_likelihood("gpu-1", &current);
        assert!(likelihood > 0.0);
    }

    #[test]
    fn test_fleet_factors() {
        let mut model = CausalModel::new();

        for _ in 0..100 {
            model.record_observation(FactorObservation {
                factor: CausalFactor::Voltage,
                normalized_value: 0.2,
                during_sdc: false,
                timestamp: Utc::now(),
                gpu_uuid: "gpu-1".to_string(),
            });
        }
        for _ in 0..10 {
            model.record_observation(FactorObservation {
                factor: CausalFactor::Voltage,
                normalized_value: 0.9,
                during_sdc: true,
                timestamp: Utc::now(),
                gpu_uuid: "gpu-2".to_string(),
            });
        }

        let fleet = model.fleet_factors();
        assert!(!fleet.is_empty());
    }
}
