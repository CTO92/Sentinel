//! Rule-based pattern detection for correlated GPU events.
//!
//! The pattern matcher operates on the temporal window for a single GPU,
//! applying a set of rules to detect known correlation patterns:
//!
//! - **Probe + Anomaly**: Probe failure + inference anomaly on same GPU within
//!   window = high-confidence SDC.
//! - **SM Localized**: Multiple probe failures on the same SM = SM-specific
//!   degradation.
//! - **Thermal Correlated**: Probe failure + thermal spike = thermal-induced
//!   transient.
//! - **Anomaly Without Probe**: Inference anomaly without any probe failure =
//!   possible application bug (not necessarily hardware SDC).
//! - **Node Correlated**: Multiple GPUs on the same node failing simultaneously.
//! - **Multi-Signal**: Multiple different anomaly types on the same GPU.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::temporal_window::{CorrelationEvent, EventType};

/// A detected correlation pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Unique identifier for this pattern instance.
    pub pattern_id: String,

    /// The type of pattern detected.
    pub pattern_type: PatternType,

    /// Confidence score for this pattern (0.0 to 1.0).
    pub confidence: f64,

    /// IDs of the events that contributed to this pattern.
    pub contributing_events: Vec<String>,

    /// GPU UUID attributed as the root cause.
    pub gpu_uuid: String,

    /// Optional SM id if the pattern is SM-localized.
    pub sm_id: Option<u32>,

    /// Human-readable description of the pattern.
    pub description: String,

    /// Recommended action based on this pattern.
    pub recommended_action: String,

    /// Severity level (1=INFO, 2=WARNING, 3=HIGH, 4=CRITICAL).
    pub severity: u32,

    /// Timestamp when the pattern was detected.
    pub detected_at: DateTime<Utc>,
}

/// Types of correlation patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    /// Probe failure + inference/training anomaly on same GPU.
    ProbeAndAnomaly,
    /// Multiple probe failures on the same SM.
    SmLocalized,
    /// Probe failure + thermal spike.
    ThermalCorrelated,
    /// Inference anomaly without probe failure.
    AnomalyWithoutProbe,
    /// Multiple GPUs on the same node failing.
    NodeCorrelated,
    /// Multiple different anomaly types on the same GPU.
    MultiSignal,
    /// TMR dissent confirming a faulty GPU.
    TmrConfirmed,
    /// Probe failure + power anomaly.
    PowerCorrelated,
    /// ECC errors coinciding with other failures.
    EccCorrelated,
}

impl std::fmt::Display for PatternType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternType::ProbeAndAnomaly => write!(f, "probe_and_anomaly"),
            PatternType::SmLocalized => write!(f, "sm_localized"),
            PatternType::ThermalCorrelated => write!(f, "thermal_correlated"),
            PatternType::AnomalyWithoutProbe => write!(f, "anomaly_without_probe"),
            PatternType::NodeCorrelated => write!(f, "node_correlated"),
            PatternType::MultiSignal => write!(f, "multi_signal"),
            PatternType::TmrConfirmed => write!(f, "tmr_confirmed"),
            PatternType::PowerCorrelated => write!(f, "power_correlated"),
            PatternType::EccCorrelated => write!(f, "ecc_correlated"),
        }
    }
}

/// Configuration for pattern matching thresholds.
#[derive(Debug, Clone)]
pub struct PatternMatcherConfig {
    /// Minimum number of probe failures on the same SM to trigger SM-localized pattern.
    pub sm_failure_threshold: usize,

    /// Minimum number of distinct anomaly types for multi-signal pattern.
    pub multi_signal_min_types: usize,

    /// Minimum number of GPUs on the same node for node-correlated pattern.
    pub node_correlated_min_gpus: usize,
}

impl Default for PatternMatcherConfig {
    fn default() -> Self {
        Self {
            sm_failure_threshold: 2,
            multi_signal_min_types: 2,
            node_correlated_min_gpus: 2,
        }
    }
}

/// The pattern matcher evaluates events within a temporal window and produces
/// detected patterns.
#[derive(Debug)]
pub struct PatternMatcher {
    config: PatternMatcherConfig,
}

impl PatternMatcher {
    /// Create a new pattern matcher with the given configuration.
    pub fn new(config: PatternMatcherConfig) -> Self {
        Self { config }
    }

    /// Create a pattern matcher with default configuration.
    pub fn with_defaults() -> Self {
        Self {
            config: PatternMatcherConfig::default(),
        }
    }

    /// Run all pattern-matching rules on the given set of events for a single GPU.
    ///
    /// `events` should be the events within the temporal window for this GPU,
    /// ordered by timestamp.
    pub fn match_patterns(&self, gpu_uuid: &str, events: &[&CorrelationEvent]) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();

        // Rule 1: Probe failure + inference/training anomaly
        if let Some(p) = self.check_probe_and_anomaly(gpu_uuid, events) {
            patterns.push(p);
        }

        // Rule 2: SM-localized failures
        patterns.extend(self.check_sm_localized(gpu_uuid, events));

        // Rule 3: Thermal correlated
        if let Some(p) = self.check_thermal_correlated(gpu_uuid, events) {
            patterns.push(p);
        }

        // Rule 4: Anomaly without probe failure
        if let Some(p) = self.check_anomaly_without_probe(gpu_uuid, events) {
            patterns.push(p);
        }

        // Rule 5: Multi-signal
        if let Some(p) = self.check_multi_signal(gpu_uuid, events) {
            patterns.push(p);
        }

        // Rule 6: Power correlated
        if let Some(p) = self.check_power_correlated(gpu_uuid, events) {
            patterns.push(p);
        }

        // Rule 7: ECC correlated
        if let Some(p) = self.check_ecc_correlated(gpu_uuid, events) {
            patterns.push(p);
        }

        patterns
    }

    /// Run the node-correlated pattern across all GPUs on the same host.
    ///
    /// `host_events` maps GPU UUID to the list of events within the window.
    pub fn match_node_patterns(
        &self,
        hostname: &str,
        host_events: &HashMap<String, Vec<&CorrelationEvent>>,
    ) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();

        // Count GPUs with failures.
        let gpus_with_failures: Vec<&String> = host_events
            .iter()
            .filter(|(_, events)| {
                events.iter().any(|e| {
                    matches!(
                        e.event_type,
                        EventType::ProbeFail | EventType::InferenceAnomaly | EventType::TrainingAnomaly
                    )
                })
            })
            .map(|(gpu, _)| gpu)
            .collect();

        if gpus_with_failures.len() >= self.config.node_correlated_min_gpus {
            let contributing: Vec<String> = host_events
                .values()
                .flat_map(|events| events.iter().map(|e| e.event_id.clone()))
                .collect();

            patterns.push(DetectedPattern {
                pattern_id: uuid::Uuid::new_v4().to_string(),
                pattern_type: PatternType::NodeCorrelated,
                confidence: 0.7,
                contributing_events: contributing,
                gpu_uuid: gpus_with_failures[0].clone(),
                sm_id: None,
                description: format!(
                    "Node-correlated failures: {} GPUs on host {} showing simultaneous failures",
                    gpus_with_failures.len(),
                    hostname
                ),
                recommended_action: "Investigate node-level issue (PSU, PCIe, motherboard)".to_string(),
                severity: 3,
                detected_at: Utc::now(),
            });
        }

        patterns
    }

    /// Check for probe failure + inference/training anomaly on the same GPU.
    fn check_probe_and_anomaly(
        &self,
        gpu_uuid: &str,
        events: &[&CorrelationEvent],
    ) -> Option<DetectedPattern> {
        let probe_failures: Vec<&CorrelationEvent> = events
            .iter()
            .filter(|e| e.event_type == EventType::ProbeFail)
            .copied()
            .collect();

        let anomalies: Vec<&CorrelationEvent> = events
            .iter()
            .filter(|e| {
                matches!(
                    e.event_type,
                    EventType::InferenceAnomaly | EventType::TrainingAnomaly | EventType::InvariantViolation
                )
            })
            .copied()
            .collect();

        if probe_failures.is_empty() || anomalies.is_empty() {
            return None;
        }

        let contributing: Vec<String> = probe_failures
            .iter()
            .chain(anomalies.iter())
            .map(|e| e.event_id.clone())
            .collect();

        Some(DetectedPattern {
            pattern_id: uuid::Uuid::new_v4().to_string(),
            pattern_type: PatternType::ProbeAndAnomaly,
            confidence: 0.95,
            contributing_events: contributing,
            gpu_uuid: gpu_uuid.to_string(),
            sm_id: probe_failures.first().and_then(|e| e.sm_id),
            description: format!(
                "High-confidence SDC: {} probe failure(s) and {} anomaly/anomalies on GPU {} within temporal window",
                probe_failures.len(),
                anomalies.len(),
                gpu_uuid
            ),
            recommended_action: "Quarantine GPU immediately".to_string(),
            severity: 4,
            detected_at: Utc::now(),
        })
    }

    /// Check for multiple probe failures localized to the same SM.
    fn check_sm_localized(
        &self,
        gpu_uuid: &str,
        events: &[&CorrelationEvent],
    ) -> Vec<DetectedPattern> {
        let mut sm_failures: HashMap<u32, Vec<&CorrelationEvent>> = HashMap::new();

        for event in events {
            if event.event_type == EventType::ProbeFail {
                if let Some(sm) = event.sm_id {
                    sm_failures.entry(sm).or_default().push(event);
                }
            }
        }

        sm_failures
            .into_iter()
            .filter(|(_, failures)| failures.len() >= self.config.sm_failure_threshold)
            .map(|(sm_id, failures)| {
                let contributing: Vec<String> =
                    failures.iter().map(|e| e.event_id.clone()).collect();

                DetectedPattern {
                    pattern_id: uuid::Uuid::new_v4().to_string(),
                    pattern_type: PatternType::SmLocalized,
                    confidence: 0.85,
                    contributing_events: contributing,
                    gpu_uuid: gpu_uuid.to_string(),
                    sm_id: Some(sm_id),
                    description: format!(
                        "SM-localized degradation: {} failures on SM {} of GPU {}",
                        failures.len(),
                        sm_id,
                        gpu_uuid
                    ),
                    recommended_action: "Consider SM-level masking or GPU quarantine".to_string(),
                    severity: 3,
                    detected_at: Utc::now(),
                }
            })
            .collect()
    }

    /// Check for probe failure coinciding with thermal spike.
    fn check_thermal_correlated(
        &self,
        gpu_uuid: &str,
        events: &[&CorrelationEvent],
    ) -> Option<DetectedPattern> {
        let has_probe_fail = events.iter().any(|e| e.event_type == EventType::ProbeFail);
        let has_thermal = events.iter().any(|e| e.event_type == EventType::ThermalSpike);

        if !has_probe_fail || !has_thermal {
            return None;
        }

        let contributing: Vec<String> = events
            .iter()
            .filter(|e| {
                e.event_type == EventType::ProbeFail || e.event_type == EventType::ThermalSpike
            })
            .map(|e| e.event_id.clone())
            .collect();

        Some(DetectedPattern {
            pattern_id: uuid::Uuid::new_v4().to_string(),
            pattern_type: PatternType::ThermalCorrelated,
            confidence: 0.7,
            contributing_events: contributing,
            gpu_uuid: gpu_uuid.to_string(),
            sm_id: None,
            description: format!(
                "Thermal-correlated failure: probe failure coincides with thermal spike on GPU {}",
                gpu_uuid
            ),
            recommended_action: "Check cooling, may be transient; increase monitoring".to_string(),
            severity: 2,
            detected_at: Utc::now(),
        })
    }

    /// Check for anomaly events without any probe failures (possible app bug).
    fn check_anomaly_without_probe(
        &self,
        gpu_uuid: &str,
        events: &[&CorrelationEvent],
    ) -> Option<DetectedPattern> {
        let has_probe_fail = events.iter().any(|e| e.event_type == EventType::ProbeFail);
        let anomalies: Vec<&CorrelationEvent> = events
            .iter()
            .filter(|e| {
                matches!(
                    e.event_type,
                    EventType::InferenceAnomaly | EventType::TrainingAnomaly | EventType::InvariantViolation
                )
            })
            .copied()
            .collect();

        if has_probe_fail || anomalies.is_empty() {
            return None;
        }

        let contributing: Vec<String> = anomalies.iter().map(|e| e.event_id.clone()).collect();

        Some(DetectedPattern {
            pattern_id: uuid::Uuid::new_v4().to_string(),
            pattern_type: PatternType::AnomalyWithoutProbe,
            confidence: 0.4,
            contributing_events: contributing,
            gpu_uuid: gpu_uuid.to_string(),
            sm_id: None,
            description: format!(
                "Anomaly without probe failure on GPU {}: {} anomaly event(s) but all probes passing. Possible application bug.",
                gpu_uuid,
                anomalies.len()
            ),
            recommended_action: "Investigate application behavior; do not quarantine GPU yet".to_string(),
            severity: 2,
            detected_at: Utc::now(),
        })
    }

    /// Check for multiple distinct anomaly types on the same GPU.
    fn check_multi_signal(
        &self,
        gpu_uuid: &str,
        events: &[&CorrelationEvent],
    ) -> Option<DetectedPattern> {
        let failure_types: std::collections::HashSet<EventType> = events
            .iter()
            .filter(|e| {
                matches!(
                    e.event_type,
                    EventType::ProbeFail
                        | EventType::InferenceAnomaly
                        | EventType::TrainingAnomaly
                        | EventType::InvariantViolation
                        | EventType::EccError
                )
            })
            .map(|e| e.event_type)
            .collect();

        if failure_types.len() < self.config.multi_signal_min_types {
            return None;
        }

        let contributing: Vec<String> = events
            .iter()
            .filter(|e| failure_types.contains(&e.event_type))
            .map(|e| e.event_id.clone())
            .collect();

        Some(DetectedPattern {
            pattern_id: uuid::Uuid::new_v4().to_string(),
            pattern_type: PatternType::MultiSignal,
            confidence: 0.9,
            contributing_events: contributing,
            gpu_uuid: gpu_uuid.to_string(),
            sm_id: None,
            description: format!(
                "Multi-signal failure: {} distinct failure types on GPU {}",
                failure_types.len(),
                gpu_uuid
            ),
            recommended_action: "Strong evidence of hardware issue; quarantine recommended".to_string(),
            severity: 4,
            detected_at: Utc::now(),
        })
    }

    /// Check for probe failure coinciding with power anomaly.
    fn check_power_correlated(
        &self,
        gpu_uuid: &str,
        events: &[&CorrelationEvent],
    ) -> Option<DetectedPattern> {
        let has_probe_fail = events.iter().any(|e| e.event_type == EventType::ProbeFail);
        let has_power = events.iter().any(|e| e.event_type == EventType::PowerAnomaly);

        if !has_probe_fail || !has_power {
            return None;
        }

        let contributing: Vec<String> = events
            .iter()
            .filter(|e| {
                e.event_type == EventType::ProbeFail || e.event_type == EventType::PowerAnomaly
            })
            .map(|e| e.event_id.clone())
            .collect();

        Some(DetectedPattern {
            pattern_id: uuid::Uuid::new_v4().to_string(),
            pattern_type: PatternType::PowerCorrelated,
            confidence: 0.65,
            contributing_events: contributing,
            gpu_uuid: gpu_uuid.to_string(),
            sm_id: None,
            description: format!(
                "Power-correlated failure on GPU {}: probe failure coincides with power anomaly",
                gpu_uuid
            ),
            recommended_action: "Check PSU and power delivery; may be transient".to_string(),
            severity: 2,
            detected_at: Utc::now(),
        })
    }

    /// Check for ECC errors coinciding with other failures.
    fn check_ecc_correlated(
        &self,
        gpu_uuid: &str,
        events: &[&CorrelationEvent],
    ) -> Option<DetectedPattern> {
        let has_ecc = events.iter().any(|e| e.event_type == EventType::EccError);
        let has_other_failure = events.iter().any(|e| {
            matches!(
                e.event_type,
                EventType::ProbeFail | EventType::InferenceAnomaly | EventType::TrainingAnomaly
            )
        });

        if !has_ecc || !has_other_failure {
            return None;
        }

        let contributing: Vec<String> = events
            .iter()
            .filter(|e| {
                matches!(
                    e.event_type,
                    EventType::EccError
                        | EventType::ProbeFail
                        | EventType::InferenceAnomaly
                        | EventType::TrainingAnomaly
                )
            })
            .map(|e| e.event_id.clone())
            .collect();

        Some(DetectedPattern {
            pattern_id: uuid::Uuid::new_v4().to_string(),
            pattern_type: PatternType::EccCorrelated,
            confidence: 0.8,
            contributing_events: contributing,
            gpu_uuid: gpu_uuid.to_string(),
            sm_id: None,
            description: format!(
                "ECC-correlated failure on GPU {}: memory errors coincide with computation failures",
                gpu_uuid
            ),
            recommended_action: "Likely memory degradation; schedule deep test".to_string(),
            severity: 3,
            detected_at: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlation::temporal_window::CorrelationEvent;

    fn make_event(gpu: &str, et: EventType, sm_id: Option<u32>) -> CorrelationEvent {
        CorrelationEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            gpu_uuid: gpu.to_string(),
            sm_id,
            event_type: et,
            timestamp: Utc::now(),
            severity: 2,
            score: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_probe_and_anomaly_pattern() {
        let matcher = PatternMatcher::with_defaults();
        let e1 = make_event("gpu-1", EventType::ProbeFail, Some(3));
        let e2 = make_event("gpu-1", EventType::InferenceAnomaly, None);
        let events: Vec<&CorrelationEvent> = vec![&e1, &e2];

        let patterns = matcher.match_patterns("gpu-1", &events);
        assert!(patterns.iter().any(|p| p.pattern_type == PatternType::ProbeAndAnomaly));
    }

    #[test]
    fn test_sm_localized_pattern() {
        let matcher = PatternMatcher::with_defaults();
        let e1 = make_event("gpu-1", EventType::ProbeFail, Some(5));
        let e2 = make_event("gpu-1", EventType::ProbeFail, Some(5));
        let events: Vec<&CorrelationEvent> = vec![&e1, &e2];

        let patterns = matcher.match_patterns("gpu-1", &events);
        assert!(patterns.iter().any(|p| p.pattern_type == PatternType::SmLocalized));
    }

    #[test]
    fn test_thermal_correlated_pattern() {
        let matcher = PatternMatcher::with_defaults();
        let e1 = make_event("gpu-1", EventType::ProbeFail, None);
        let e2 = make_event("gpu-1", EventType::ThermalSpike, None);
        let events: Vec<&CorrelationEvent> = vec![&e1, &e2];

        let patterns = matcher.match_patterns("gpu-1", &events);
        assert!(patterns.iter().any(|p| p.pattern_type == PatternType::ThermalCorrelated));
    }

    #[test]
    fn test_anomaly_without_probe() {
        let matcher = PatternMatcher::with_defaults();
        let e1 = make_event("gpu-1", EventType::InferenceAnomaly, None);
        let events: Vec<&CorrelationEvent> = vec![&e1];

        let patterns = matcher.match_patterns("gpu-1", &events);
        assert!(patterns.iter().any(|p| p.pattern_type == PatternType::AnomalyWithoutProbe));
    }

    #[test]
    fn test_no_pattern_for_pass_only() {
        let matcher = PatternMatcher::with_defaults();
        let e1 = make_event("gpu-1", EventType::ProbePass, None);
        let events: Vec<&CorrelationEvent> = vec![&e1];

        let patterns = matcher.match_patterns("gpu-1", &events);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_multi_signal_pattern() {
        let matcher = PatternMatcher::with_defaults();
        let e1 = make_event("gpu-1", EventType::ProbeFail, None);
        let e2 = make_event("gpu-1", EventType::InferenceAnomaly, None);
        let e3 = make_event("gpu-1", EventType::EccError, None);
        let events: Vec<&CorrelationEvent> = vec![&e1, &e2, &e3];

        let patterns = matcher.match_patterns("gpu-1", &events);
        assert!(patterns.iter().any(|p| p.pattern_type == PatternType::MultiSignal));
    }
}
