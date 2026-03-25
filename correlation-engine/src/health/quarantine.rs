//! GPU quarantine state machine.
//!
//! Implements the lifecycle state transitions:
//!
//! ```text
//! HEALTHY -> SUSPECT -> QUARANTINED -> DEEP_TEST -> CONDEMNED
//!   ^          |            |              |
//!   |          |            |              |
//!   +----------+            +--------------+
//!  (cleared)             (passed -> HEALTHY)
//! ```
//!
//! All transitions are validated against the state machine rules and recorded
//! with timestamps and reasons for audit trail purposes.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::util::config::QuarantineConfig;

/// Errors in quarantine state transitions.
#[derive(Debug, Error)]
pub enum QuarantineError {
    #[error("invalid state transition from {from:?} via {action:?}")]
    InvalidTransition {
        from: QuarantineState,
        action: QuarantineAction,
    },

    #[error("GPU {gpu_uuid} not found in quarantine manager")]
    GpuNotFound { gpu_uuid: String },

    #[error("action {action:?} requires approval for GPU {gpu_uuid}")]
    ApprovalRequired {
        action: QuarantineAction,
        gpu_uuid: String,
    },
}

/// Lifecycle states for a GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuarantineState {
    /// GPU is operating normally.
    Healthy,
    /// GPU has shown anomalous signals; under increased monitoring.
    Suspect,
    /// GPU has been removed from production workloads.
    Quarantined,
    /// GPU is undergoing deep diagnostic testing.
    DeepTest,
    /// GPU has been permanently marked as unreliable.
    Condemned,
}

impl std::fmt::Display for QuarantineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuarantineState::Healthy => write!(f, "HEALTHY"),
            QuarantineState::Suspect => write!(f, "SUSPECT"),
            QuarantineState::Quarantined => write!(f, "QUARANTINED"),
            QuarantineState::DeepTest => write!(f, "DEEP_TEST"),
            QuarantineState::Condemned => write!(f, "CONDEMNED"),
        }
    }
}

/// Actions that can be applied to a GPU's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuarantineAction {
    /// Mark GPU as suspect (from Healthy).
    MarkSuspect,
    /// Quarantine the GPU (from Suspect).
    Quarantine,
    /// Schedule deep diagnostic testing (from Quarantined).
    ScheduleDeepTest,
    /// Condemn the GPU permanently.
    Condemn,
    /// Reinstate the GPU to Healthy (from Suspect, DeepTest).
    Reinstate,
    /// Clear suspect status (from Suspect -> Healthy).
    ClearSuspect,
}

impl std::fmt::Display for QuarantineAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuarantineAction::MarkSuspect => write!(f, "mark_suspect"),
            QuarantineAction::Quarantine => write!(f, "quarantine"),
            QuarantineAction::ScheduleDeepTest => write!(f, "schedule_deep_test"),
            QuarantineAction::Condemn => write!(f, "condemn"),
            QuarantineAction::Reinstate => write!(f, "reinstate"),
            QuarantineAction::ClearSuspect => write!(f, "clear_suspect"),
        }
    }
}

/// A recorded state transition for audit purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransitionRecord {
    /// GPU UUID.
    pub gpu_uuid: String,
    /// State before the transition.
    pub from_state: QuarantineState,
    /// State after the transition.
    pub to_state: QuarantineState,
    /// Action that caused the transition.
    pub action: QuarantineAction,
    /// Human-readable reason.
    pub reason: String,
    /// When the transition occurred.
    pub timestamp: DateTime<Utc>,
}

/// Per-GPU state record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStateRecord {
    /// Current lifecycle state.
    pub state: QuarantineState,
    /// When the GPU entered this state.
    pub state_entered_at: DateTime<Utc>,
    /// Reason for the most recent transition.
    pub reason: String,
    /// Number of consecutive deep-test passes (used for reinstatement).
    pub deep_test_passes: u64,
    /// Full transition history.
    pub history: Vec<StateTransitionRecord>,
}

impl GpuStateRecord {
    fn new() -> Self {
        Self {
            state: QuarantineState::Healthy,
            state_entered_at: Utc::now(),
            reason: "initial state".to_string(),
            deep_test_passes: 0,
            history: Vec::new(),
        }
    }
}

/// Manages the quarantine lifecycle for all GPUs in the fleet.
#[derive(Debug)]
pub struct QuarantineManager {
    /// Configuration.
    config: QuarantineConfig,

    /// Per-GPU state records.
    states: HashMap<String, GpuStateRecord>,
}

impl QuarantineManager {
    /// Create a new quarantine manager with the given configuration.
    pub fn new(config: QuarantineConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
        }
    }

    /// Get the current state of a GPU. Returns `Healthy` for unknown GPUs.
    pub fn get_state(&self, gpu_uuid: &str) -> QuarantineState {
        self.states
            .get(gpu_uuid)
            .map(|r| r.state)
            .unwrap_or(QuarantineState::Healthy)
    }

    /// Get the full state record for a GPU.
    pub fn get_record(&self, gpu_uuid: &str) -> Option<&GpuStateRecord> {
        self.states.get(gpu_uuid)
    }

    /// Get the transition history for a GPU.
    pub fn get_history(&self, gpu_uuid: &str) -> Vec<StateTransitionRecord> {
        self.states
            .get(gpu_uuid)
            .map(|r| r.history.clone())
            .unwrap_or_default()
    }

    /// Attempt a state transition. Returns the new state on success.
    pub fn transition(
        &mut self,
        gpu_uuid: &str,
        action: QuarantineAction,
        reason: &str,
    ) -> Result<QuarantineState, QuarantineError> {
        let current = self
            .states
            .get(gpu_uuid)
            .map(|r| r.state)
            .unwrap_or(QuarantineState::Healthy);

        let new_state = self.validate_transition(current, action)?;

        let record = self
            .states
            .entry(gpu_uuid.to_string())
            .or_insert_with(GpuStateRecord::new);

        // Check if condemn requires approval.
        if action == QuarantineAction::Condemn && self.config.condemn_requires_approval {
            // In production, this would be gated by an approval workflow.
            // For now, we log a warning and proceed.
            tracing::warn!(
                gpu = %gpu_uuid,
                "Condemn action proceeding without explicit approval (approval workflow not connected)"
            );
        }

        let transition = StateTransitionRecord {
            gpu_uuid: gpu_uuid.to_string(),
            from_state: current,
            to_state: new_state,
            action,
            reason: reason.to_string(),
            timestamp: Utc::now(),
        };

        record.state = new_state;
        record.state_entered_at = Utc::now();
        record.reason = reason.to_string();
        record.history.push(transition);

        // Reset deep-test pass counter on certain transitions.
        if new_state != QuarantineState::DeepTest {
            record.deep_test_passes = 0;
        }

        Ok(new_state)
    }

    /// Record a successful deep-test probe. Returns `Some(Healthy)` if the GPU
    /// has passed enough tests to be reinstated.
    pub fn record_deep_test_pass(
        &mut self,
        gpu_uuid: &str,
    ) -> Result<Option<QuarantineState>, QuarantineError> {
        let record = self
            .states
            .get_mut(gpu_uuid)
            .ok_or_else(|| QuarantineError::GpuNotFound {
                gpu_uuid: gpu_uuid.to_string(),
            })?;

        if record.state != QuarantineState::DeepTest {
            return Ok(None);
        }

        record.deep_test_passes += 1;

        if record.deep_test_passes >= self.config.reinstatement_pass_count {
            let transition = StateTransitionRecord {
                gpu_uuid: gpu_uuid.to_string(),
                from_state: QuarantineState::DeepTest,
                to_state: QuarantineState::Healthy,
                action: QuarantineAction::Reinstate,
                reason: format!(
                    "Passed {} consecutive deep-test probes",
                    record.deep_test_passes
                ),
                timestamp: Utc::now(),
            };
            record.state = QuarantineState::Healthy;
            record.state_entered_at = Utc::now();
            record.reason = transition.reason.clone();
            record.deep_test_passes = 0;
            record.history.push(transition);
            Ok(Some(QuarantineState::Healthy))
        } else {
            Ok(None)
        }
    }

    /// Validate whether a transition is legal according to the state machine.
    fn validate_transition(
        &self,
        current: QuarantineState,
        action: QuarantineAction,
    ) -> Result<QuarantineState, QuarantineError> {
        match (current, action) {
            // HEALTHY transitions
            (QuarantineState::Healthy, QuarantineAction::MarkSuspect) => {
                Ok(QuarantineState::Suspect)
            }
            (QuarantineState::Healthy, QuarantineAction::Quarantine) => {
                Ok(QuarantineState::Quarantined)
            }
            (QuarantineState::Healthy, QuarantineAction::Condemn) => {
                Ok(QuarantineState::Condemned)
            }

            // SUSPECT transitions
            (QuarantineState::Suspect, QuarantineAction::Quarantine) => {
                Ok(QuarantineState::Quarantined)
            }
            (QuarantineState::Suspect, QuarantineAction::ClearSuspect) => {
                Ok(QuarantineState::Healthy)
            }
            (QuarantineState::Suspect, QuarantineAction::Reinstate) => {
                Ok(QuarantineState::Healthy)
            }
            (QuarantineState::Suspect, QuarantineAction::Condemn) => {
                Ok(QuarantineState::Condemned)
            }

            // QUARANTINED transitions
            (QuarantineState::Quarantined, QuarantineAction::ScheduleDeepTest) => {
                Ok(QuarantineState::DeepTest)
            }
            (QuarantineState::Quarantined, QuarantineAction::Condemn) => {
                Ok(QuarantineState::Condemned)
            }
            (QuarantineState::Quarantined, QuarantineAction::Reinstate) => {
                Ok(QuarantineState::Healthy)
            }

            // DEEP_TEST transitions
            (QuarantineState::DeepTest, QuarantineAction::Reinstate) => {
                Ok(QuarantineState::Healthy)
            }
            (QuarantineState::DeepTest, QuarantineAction::Condemn) => {
                Ok(QuarantineState::Condemned)
            }

            // CONDEMNED is a terminal state (no transitions out).
            (QuarantineState::Condemned, _) => Err(QuarantineError::InvalidTransition {
                from: current,
                action,
            }),

            // Any other combination is invalid.
            _ => Err(QuarantineError::InvalidTransition {
                from: current,
                action,
            }),
        }
    }

    /// Get all GPUs in a specific state.
    pub fn gpus_in_state(&self, state: QuarantineState) -> Vec<String> {
        self.states
            .iter()
            .filter(|(_, r)| r.state == state)
            .map(|(uuid, _)| uuid.clone())
            .collect()
    }

    /// Get state counts for all lifecycle states.
    pub fn state_counts(&self) -> HashMap<QuarantineState, usize> {
        let mut counts = HashMap::new();
        counts.insert(QuarantineState::Healthy, 0);
        counts.insert(QuarantineState::Suspect, 0);
        counts.insert(QuarantineState::Quarantined, 0);
        counts.insert(QuarantineState::DeepTest, 0);
        counts.insert(QuarantineState::Condemned, 0);

        for record in self.states.values() {
            *counts.entry(record.state).or_insert(0) += 1;
        }
        counts
    }

    /// Get the total number of tracked GPUs.
    pub fn gpu_count(&self) -> usize {
        self.states.len()
    }

    /// Restore a GPU's state (e.g., from Redis on startup).
    pub fn restore_state(&mut self, gpu_uuid: &str, record: GpuStateRecord) {
        self.states.insert(gpu_uuid.to_string(), record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager() -> QuarantineManager {
        QuarantineManager::new(QuarantineConfig {
            reinstatement_pass_count: 3,
            ..QuarantineConfig::default()
        })
    }

    #[test]
    fn test_initial_state_is_healthy() {
        let qm = test_manager();
        assert_eq!(qm.get_state("gpu-1"), QuarantineState::Healthy);
    }

    #[test]
    fn test_healthy_to_suspect() {
        let mut qm = test_manager();
        let result = qm.transition("gpu-1", QuarantineAction::MarkSuspect, "test");
        assert_eq!(result.unwrap(), QuarantineState::Suspect);
    }

    #[test]
    fn test_suspect_to_quarantined() {
        let mut qm = test_manager();
        qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
            .unwrap();
        let result = qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine");
        assert_eq!(result.unwrap(), QuarantineState::Quarantined);
    }

    #[test]
    fn test_quarantined_to_deep_test() {
        let mut qm = test_manager();
        qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
            .unwrap();
        qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine")
            .unwrap();
        let result = qm.transition("gpu-1", QuarantineAction::ScheduleDeepTest, "deep test");
        assert_eq!(result.unwrap(), QuarantineState::DeepTest);
    }

    #[test]
    fn test_deep_test_reinstatement() {
        let mut qm = test_manager();
        qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
            .unwrap();
        qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine")
            .unwrap();
        qm.transition("gpu-1", QuarantineAction::ScheduleDeepTest, "deep test")
            .unwrap();

        // Pass 2 tests - not enough.
        assert!(qm.record_deep_test_pass("gpu-1").unwrap().is_none());
        assert!(qm.record_deep_test_pass("gpu-1").unwrap().is_none());

        // Pass 3rd test - should reinstate.
        let result = qm.record_deep_test_pass("gpu-1").unwrap();
        assert_eq!(result, Some(QuarantineState::Healthy));
        assert_eq!(qm.get_state("gpu-1"), QuarantineState::Healthy);
    }

    #[test]
    fn test_condemned_is_terminal() {
        let mut qm = test_manager();
        qm.transition("gpu-1", QuarantineAction::Condemn, "bad gpu")
            .unwrap();
        assert!(qm
            .transition("gpu-1", QuarantineAction::Reinstate, "try to reinstate")
            .is_err());
    }

    #[test]
    fn test_suspect_cleared() {
        let mut qm = test_manager();
        qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
            .unwrap();
        let result = qm.transition("gpu-1", QuarantineAction::ClearSuspect, "cleared");
        assert_eq!(result.unwrap(), QuarantineState::Healthy);
    }

    #[test]
    fn test_invalid_transition() {
        let mut qm = test_manager();
        // Can't schedule deep test directly from Healthy.
        let result = qm.transition("gpu-1", QuarantineAction::ScheduleDeepTest, "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_history_tracking() {
        let mut qm = test_manager();
        qm.transition("gpu-1", QuarantineAction::MarkSuspect, "step 1")
            .unwrap();
        qm.transition("gpu-1", QuarantineAction::Quarantine, "step 2")
            .unwrap();

        let history = qm.get_history("gpu-1");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].from_state, QuarantineState::Healthy);
        assert_eq!(history[0].to_state, QuarantineState::Suspect);
        assert_eq!(history[1].from_state, QuarantineState::Suspect);
        assert_eq!(history[1].to_state, QuarantineState::Quarantined);
    }

    #[test]
    fn test_state_counts() {
        let mut qm = test_manager();
        qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
            .unwrap();
        qm.transition("gpu-2", QuarantineAction::Condemn, "bad")
            .unwrap();

        let counts = qm.state_counts();
        assert_eq!(counts[&QuarantineState::Suspect], 1);
        assert_eq!(counts[&QuarantineState::Condemned], 1);
    }
}
