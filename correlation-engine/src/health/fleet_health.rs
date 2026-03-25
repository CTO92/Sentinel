//! Fleet-wide health aggregation and summary computation.
//!
//! This module provides fleet-level views of GPU health, including summary
//! statistics, worst-performing GPUs, and cohort analysis by model, firmware
//! version, or node.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::health::gpu_health::GpuHealthRecord;
use crate::health::quarantine::QuarantineState;

/// Aggregated fleet health summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetSummary {
    /// Total number of GPUs in the fleet.
    pub total_gpus: u32,

    /// Counts by state.
    pub healthy: u32,
    pub suspect: u32,
    pub quarantined: u32,
    pub deep_test: u32,
    pub condemned: u32,

    /// Fleet-wide average reliability score.
    pub average_reliability: f64,

    /// Minimum reliability score in the fleet.
    pub min_reliability: f64,

    /// Estimated fleet SDC rate (events per GPU-hour).
    pub sdc_rate: f64,

    /// Number of active probe agents.
    pub active_agents: u32,

    /// Timestamp of this summary.
    pub snapshot_time: DateTime<Utc>,
}

/// Compute a fleet summary from a collection of GPU health records.
pub fn compute_fleet_summary(records: &[&GpuHealthRecord]) -> FleetSummary {
    let total = records.len() as u32;
    let mut healthy = 0u32;
    let mut suspect = 0u32;
    let mut quarantined = 0u32;
    let mut deep_test = 0u32;
    let mut condemned = 0u32;
    let mut reliability_sum = 0.0f64;
    let mut min_reliability = 1.0f64;
    let mut sdc_rate_sum = 0.0f64;
    let mut hostnames = std::collections::HashSet::new();

    for record in records {
        match record.state {
            QuarantineState::Healthy => healthy += 1,
            QuarantineState::Suspect => suspect += 1,
            QuarantineState::Quarantined => quarantined += 1,
            QuarantineState::DeepTest => deep_test += 1,
            QuarantineState::Condemned => condemned += 1,
        }

        let score = record.reliability_score();
        reliability_sum += score;
        if score < min_reliability {
            min_reliability = score;
        }

        sdc_rate_sum += record.probe_failure_rate;
        hostnames.insert(record.hostname.clone());
    }

    let average_reliability = if total > 0 {
        reliability_sum / total as f64
    } else {
        1.0
    };

    let sdc_rate = if total > 0 {
        sdc_rate_sum / total as f64
    } else {
        0.0
    };

    FleetSummary {
        total_gpus: total,
        healthy,
        suspect,
        quarantined,
        deep_test,
        condemned,
        average_reliability,
        min_reliability: if total > 0 { min_reliability } else { 1.0 },
        sdc_rate,
        active_agents: hostnames.len() as u32,
        snapshot_time: Utc::now(),
    }
}

/// Cohort analysis: group GPUs by a common attribute and compare health metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohortSummary {
    /// The cohort attribute value (e.g., firmware version, model name).
    pub cohort_key: String,

    /// Number of GPUs in this cohort.
    pub count: u32,

    /// Average reliability score.
    pub average_reliability: f64,

    /// Number of GPUs not in Healthy state.
    pub unhealthy_count: u32,

    /// Average probe failure rate.
    pub average_failure_rate: f64,
}

/// Group GPUs by model and compute per-cohort statistics.
pub fn cohort_by_model(records: &[&GpuHealthRecord]) -> Vec<CohortSummary> {
    let mut groups: HashMap<String, Vec<&GpuHealthRecord>> = HashMap::new();
    for record in records {
        groups
            .entry(record.model.clone())
            .or_default()
            .push(record);
    }

    groups
        .into_iter()
        .map(|(model, recs)| compute_cohort(&model, &recs))
        .collect()
}

/// Group GPUs by firmware version and compute per-cohort statistics.
pub fn cohort_by_firmware(records: &[&GpuHealthRecord]) -> Vec<CohortSummary> {
    let mut groups: HashMap<String, Vec<&GpuHealthRecord>> = HashMap::new();
    for record in records {
        groups
            .entry(record.firmware_version.clone())
            .or_default()
            .push(record);
    }

    groups
        .into_iter()
        .map(|(fw, recs)| compute_cohort(&fw, &recs))
        .collect()
}

/// Group GPUs by hostname and compute per-cohort statistics.
pub fn cohort_by_node(records: &[&GpuHealthRecord]) -> Vec<CohortSummary> {
    let mut groups: HashMap<String, Vec<&GpuHealthRecord>> = HashMap::new();
    for record in records {
        groups
            .entry(record.hostname.clone())
            .or_default()
            .push(record);
    }

    groups
        .into_iter()
        .map(|(host, recs)| compute_cohort(&host, &recs))
        .collect()
}

fn compute_cohort(key: &str, records: &[&GpuHealthRecord]) -> CohortSummary {
    let count = records.len() as u32;
    let reliability_sum: f64 = records.iter().map(|r| r.reliability_score()).sum();
    let unhealthy = records
        .iter()
        .filter(|r| r.state != QuarantineState::Healthy)
        .count() as u32;
    let failure_rate_sum: f64 = records.iter().map(|r| r.probe_failure_rate).sum();

    CohortSummary {
        cohort_key: key.to_string(),
        count,
        average_reliability: if count > 0 {
            reliability_sum / count as f64
        } else {
            1.0
        },
        unhealthy_count: unhealthy,
        average_failure_rate: if count > 0 {
            failure_rate_sum / count as f64
        } else {
            0.0
        },
    }
}

/// Get the N worst-performing GPUs by reliability score.
pub fn worst_gpus<'a>(records: &[&'a GpuHealthRecord], n: usize) -> Vec<&'a GpuHealthRecord> {
    let mut sorted: Vec<&GpuHealthRecord> = records.to_vec();
    sorted.sort_by(|a, b| {
        a.reliability_score()
            .partial_cmp(&b.reliability_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.into_iter().take(n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlation::bayesian_attribution::BayesianBelief;

    fn make_record(uuid: &str, model: &str, state: QuarantineState) -> GpuHealthRecord {
        let mut rec = GpuHealthRecord::new(
            uuid.to_string(),
            "host-1".to_string(),
            0,
            model.to_string(),
            1000.0,
            1.0,
        );
        rec.state = state;
        rec
    }

    #[test]
    fn test_fleet_summary() {
        let r1 = make_record("gpu-1", "H100", QuarantineState::Healthy);
        let r2 = make_record("gpu-2", "H100", QuarantineState::Suspect);
        let r3 = make_record("gpu-3", "A100", QuarantineState::Quarantined);

        let records: Vec<&GpuHealthRecord> = vec![&r1, &r2, &r3];
        let summary = compute_fleet_summary(&records);

        assert_eq!(summary.total_gpus, 3);
        assert_eq!(summary.healthy, 1);
        assert_eq!(summary.suspect, 1);
        assert_eq!(summary.quarantined, 1);
    }

    #[test]
    fn test_cohort_by_model() {
        let r1 = make_record("gpu-1", "H100", QuarantineState::Healthy);
        let r2 = make_record("gpu-2", "H100", QuarantineState::Healthy);
        let r3 = make_record("gpu-3", "A100", QuarantineState::Suspect);

        let records: Vec<&GpuHealthRecord> = vec![&r1, &r2, &r3];
        let cohorts = cohort_by_model(&records);

        assert_eq!(cohorts.len(), 2);
        let h100_cohort = cohorts.iter().find(|c| c.cohort_key == "H100").unwrap();
        assert_eq!(h100_cohort.count, 2);
        assert_eq!(h100_cohort.unhealthy_count, 0);
    }

    #[test]
    fn test_worst_gpus() {
        let mut r1 = make_record("gpu-1", "H100", QuarantineState::Healthy);
        let mut r2 = make_record("gpu-2", "H100", QuarantineState::Healthy);
        // Make gpu-2 worse by increasing its beta.
        r2.belief = BayesianBelief::new(1000.0, 10.0);

        let records: Vec<&GpuHealthRecord> = vec![&r1, &r2];
        let worst = worst_gpus(&records, 1);
        assert_eq!(worst[0].gpu_uuid, "gpu-2");
    }
}
