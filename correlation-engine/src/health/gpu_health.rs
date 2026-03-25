//! GPU-level health tracking actor.
//!
//! Each GPU has a logical "actor" that processes events sequentially for that
//! GPU. This module provides the per-GPU health aggregation that combines
//! Bayesian scores, quarantine state, and environmental factors into a
//! unified health view.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::correlation::bayesian_attribution::BayesianBelief;
use crate::health::quarantine::QuarantineState;

/// Comprehensive health record for a single GPU, combining all health signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuHealthRecord {
    /// GPU UUID.
    pub gpu_uuid: String,

    /// Hostname of the machine containing this GPU.
    pub hostname: String,

    /// PCI device index on the host.
    pub device_index: u32,

    /// GPU model name.
    pub model: String,

    /// Current lifecycle state.
    pub state: QuarantineState,

    /// Bayesian reliability belief.
    pub belief: BayesianBelief,

    /// Timestamp of the most recent probe execution.
    pub last_probe_time: Option<DateTime<Utc>>,

    /// Timestamp of the most recent anomaly.
    pub last_anomaly_time: Option<DateTime<Utc>>,

    /// Rolling probe failure rate (failures per hour).
    pub probe_failure_rate: f64,

    /// Rolling anomaly rate (anomalies per hour).
    pub anomaly_rate: f64,

    /// Per-SM health information.
    pub sm_health: HashMap<u32, SmHealthRecord>,

    /// Last known temperature (Celsius).
    pub last_temperature_c: Option<f32>,

    /// Last known power draw (Watts).
    pub last_power_w: Option<f32>,

    /// Driver version.
    pub driver_version: String,

    /// Firmware version.
    pub firmware_version: String,
}

impl GpuHealthRecord {
    /// Create a new GPU health record with default values.
    pub fn new(
        gpu_uuid: String,
        hostname: String,
        device_index: u32,
        model: String,
        prior_alpha: f64,
        prior_beta: f64,
    ) -> Self {
        Self {
            gpu_uuid,
            hostname,
            device_index,
            model,
            state: QuarantineState::Healthy,
            belief: BayesianBelief::new(prior_alpha, prior_beta),
            last_probe_time: None,
            last_anomaly_time: None,
            probe_failure_rate: 0.0,
            anomaly_rate: 0.0,
            sm_health: HashMap::new(),
            last_temperature_c: None,
            last_power_w: None,
            driver_version: String::new(),
            firmware_version: String::new(),
        }
    }

    /// Get the reliability score.
    pub fn reliability_score(&self) -> f64 {
        self.belief.reliability_score()
    }

    /// Check if the GPU is currently available for production workloads.
    pub fn is_available(&self) -> bool {
        self.state == QuarantineState::Healthy
    }
}

/// Health record for a single Streaming Multiprocessor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmHealthRecord {
    /// SM index.
    pub sm_id: u32,

    /// Bayesian belief for this SM.
    pub belief: BayesianBelief,

    /// Whether this SM is currently disabled/masked.
    pub disabled: bool,

    /// Reason for disabling.
    pub disable_reason: Option<String>,

    /// Last probe time on this SM.
    pub last_probe_time: Option<DateTime<Utc>>,
}

/// Rolling rate calculator using a sliding window of timestamps.
#[derive(Debug, Clone)]
pub struct RollingRateCalculator {
    /// Window duration in seconds.
    window_secs: f64,

    /// Timestamps of events within the window.
    timestamps: Vec<DateTime<Utc>>,
}

impl RollingRateCalculator {
    /// Create a new rate calculator with the given window duration.
    pub fn new(window_secs: f64) -> Self {
        Self {
            window_secs,
            timestamps: Vec::new(),
        }
    }

    /// Record an event at the current time.
    pub fn record(&mut self) {
        self.record_at(Utc::now());
    }

    /// Record an event at a specific time.
    pub fn record_at(&mut self, timestamp: DateTime<Utc>) {
        self.timestamps.push(timestamp);
        self.expire();
    }

    /// Compute the current rate (events per hour).
    pub fn rate_per_hour(&mut self) -> f64 {
        self.expire();
        let count = self.timestamps.len() as f64;
        let hours = self.window_secs / 3600.0;
        if hours == 0.0 {
            return 0.0;
        }
        count / hours
    }

    /// Expire events outside the window.
    fn expire(&mut self) {
        let cutoff = Utc::now()
            - chrono::Duration::milliseconds((self.window_secs * 1000.0) as i64);
        self.timestamps.retain(|t| *t >= cutoff);
    }

    /// Get the number of events in the current window.
    pub fn count(&self) -> usize {
        self.timestamps.len()
    }
}

/// Tracks GPU health for the fleet, providing a registry of health records.
#[derive(Debug)]
pub struct GpuHealthTracker {
    /// Per-GPU health records, keyed by GPU UUID.
    records: HashMap<String, GpuHealthRecord>,

    /// Probe failure rate calculators per GPU.
    probe_failure_rates: HashMap<String, RollingRateCalculator>,

    /// Anomaly rate calculators per GPU.
    anomaly_rates: HashMap<String, RollingRateCalculator>,

    /// Rolling window size in seconds for rate calculation.
    rate_window_secs: f64,

    /// Bayesian prior alpha.
    prior_alpha: f64,

    /// Bayesian prior beta.
    prior_beta: f64,
}

impl GpuHealthTracker {
    /// Create a new GPU health tracker.
    pub fn new(prior_alpha: f64, prior_beta: f64, rate_window_secs: f64) -> Self {
        Self {
            records: HashMap::new(),
            probe_failure_rates: HashMap::new(),
            anomaly_rates: HashMap::new(),
            rate_window_secs,
            prior_alpha,
            prior_beta,
        }
    }

    /// Register a GPU or update its metadata.
    pub fn register_gpu(
        &mut self,
        gpu_uuid: &str,
        hostname: &str,
        device_index: u32,
        model: &str,
    ) {
        self.records
            .entry(gpu_uuid.to_string())
            .or_insert_with(|| {
                GpuHealthRecord::new(
                    gpu_uuid.to_string(),
                    hostname.to_string(),
                    device_index,
                    model.to_string(),
                    self.prior_alpha,
                    self.prior_beta,
                )
            });
    }

    /// Get a GPU's health record.
    pub fn get_record(&self, gpu_uuid: &str) -> Option<&GpuHealthRecord> {
        self.records.get(gpu_uuid)
    }

    /// Get a mutable reference to a GPU's health record.
    pub fn get_record_mut(&mut self, gpu_uuid: &str) -> Option<&mut GpuHealthRecord> {
        self.records.get_mut(gpu_uuid)
    }

    /// Record a probe failure and update the rolling rate.
    pub fn record_probe_failure(&mut self, gpu_uuid: &str) {
        let calc = self
            .probe_failure_rates
            .entry(gpu_uuid.to_string())
            .or_insert_with(|| RollingRateCalculator::new(self.rate_window_secs));
        calc.record();

        if let Some(record) = self.records.get_mut(gpu_uuid) {
            record.probe_failure_rate = calc.rate_per_hour();
        }
    }

    /// Record an anomaly and update the rolling rate.
    pub fn record_anomaly(&mut self, gpu_uuid: &str) {
        let calc = self
            .anomaly_rates
            .entry(gpu_uuid.to_string())
            .or_insert_with(|| RollingRateCalculator::new(self.rate_window_secs));
        calc.record();

        if let Some(record) = self.records.get_mut(gpu_uuid) {
            record.anomaly_rate = calc.rate_per_hour();
        }
    }

    /// Update telemetry data for a GPU.
    pub fn update_telemetry(
        &mut self,
        gpu_uuid: &str,
        temperature_c: Option<f32>,
        power_w: Option<f32>,
    ) {
        if let Some(record) = self.records.get_mut(gpu_uuid) {
            if let Some(temp) = temperature_c {
                record.last_temperature_c = Some(temp);
            }
            if let Some(power) = power_w {
                record.last_power_w = Some(power);
            }
        }
    }

    /// Get all GPU UUIDs.
    pub fn all_gpu_uuids(&self) -> Vec<String> {
        self.records.keys().cloned().collect()
    }

    /// Get the number of tracked GPUs.
    pub fn gpu_count(&self) -> usize {
        self.records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_rate_calculator() {
        let mut calc = RollingRateCalculator::new(3600.0); // 1 hour window
        for _ in 0..10 {
            calc.record();
        }
        let rate = calc.rate_per_hour();
        assert_eq!(rate, 10.0);
    }

    #[test]
    fn test_gpu_health_tracker() {
        let mut tracker = GpuHealthTracker::new(1000.0, 1.0, 3600.0);
        tracker.register_gpu("gpu-1", "host-1", 0, "H100");

        assert!(tracker.get_record("gpu-1").is_some());
        assert!(tracker.get_record("gpu-1").unwrap().is_available());

        tracker.update_telemetry("gpu-1", Some(72.0), Some(350.0));
        let record = tracker.get_record("gpu-1").unwrap();
        assert_eq!(record.last_temperature_c, Some(72.0));
        assert_eq!(record.last_power_w, Some(350.0));
    }
}
