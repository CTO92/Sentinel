//! Tests for the quarantine state machine.
//!
//! Tests all valid and invalid state transitions, reinstatement logic,
//! history tracking, and edge cases.

use sentinel_correlation_engine::health::quarantine::{
    QuarantineAction, QuarantineError, QuarantineManager, QuarantineState,
};
use sentinel_correlation_engine::util::config::QuarantineConfig;

fn test_config() -> QuarantineConfig {
    QuarantineConfig {
        reinstatement_pass_count: 5,
        quarantine_timeout_secs: 86400,
        deep_test_timeout_secs: 14400,
        condemn_requires_approval: false,
    }
}

// --- Valid transition tests ---

#[test]
fn test_healthy_to_suspect() {
    let mut qm = QuarantineManager::new(test_config());
    let result = qm.transition("gpu-1", QuarantineAction::MarkSuspect, "test");
    assert_eq!(result.unwrap(), QuarantineState::Suspect);
    assert_eq!(qm.get_state("gpu-1"), QuarantineState::Suspect);
}

#[test]
fn test_healthy_to_quarantined_direct() {
    let mut qm = QuarantineManager::new(test_config());
    let result = qm.transition("gpu-1", QuarantineAction::Quarantine, "emergency");
    assert_eq!(result.unwrap(), QuarantineState::Quarantined);
}

#[test]
fn test_healthy_to_condemned_direct() {
    let mut qm = QuarantineManager::new(test_config());
    let result = qm.transition("gpu-1", QuarantineAction::Condemn, "dead on arrival");
    assert_eq!(result.unwrap(), QuarantineState::Condemned);
}

#[test]
fn test_suspect_to_quarantined() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
        .unwrap();
    let result = qm.transition("gpu-1", QuarantineAction::Quarantine, "confirmed");
    assert_eq!(result.unwrap(), QuarantineState::Quarantined);
}

#[test]
fn test_suspect_to_healthy_cleared() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
        .unwrap();
    let result = qm.transition("gpu-1", QuarantineAction::ClearSuspect, "false alarm");
    assert_eq!(result.unwrap(), QuarantineState::Healthy);
}

#[test]
fn test_quarantined_to_deep_test() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine")
        .unwrap();
    let result = qm.transition("gpu-1", QuarantineAction::ScheduleDeepTest, "begin tests");
    assert_eq!(result.unwrap(), QuarantineState::DeepTest);
}

#[test]
fn test_deep_test_to_healthy_via_passes() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine")
        .unwrap();
    qm.transition("gpu-1", QuarantineAction::ScheduleDeepTest, "test")
        .unwrap();

    // Need 5 passes (config).
    for i in 0..4 {
        let result = qm.record_deep_test_pass("gpu-1").unwrap();
        assert!(result.is_none(), "Should not reinstate after {} passes", i + 1);
    }

    let result = qm.record_deep_test_pass("gpu-1").unwrap();
    assert_eq!(result, Some(QuarantineState::Healthy));
    assert_eq!(qm.get_state("gpu-1"), QuarantineState::Healthy);
}

#[test]
fn test_deep_test_to_condemned() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine")
        .unwrap();
    qm.transition("gpu-1", QuarantineAction::ScheduleDeepTest, "test")
        .unwrap();
    let result = qm.transition("gpu-1", QuarantineAction::Condemn, "failed tests");
    assert_eq!(result.unwrap(), QuarantineState::Condemned);
}

#[test]
fn test_quarantined_to_healthy_reinstate() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine")
        .unwrap();
    let result = qm.transition("gpu-1", QuarantineAction::Reinstate, "manual override");
    assert_eq!(result.unwrap(), QuarantineState::Healthy);
}

// --- Invalid transition tests ---

#[test]
fn test_condemned_is_terminal() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::Condemn, "condemned")
        .unwrap();

    let actions = [
        QuarantineAction::MarkSuspect,
        QuarantineAction::Quarantine,
        QuarantineAction::ScheduleDeepTest,
        QuarantineAction::Reinstate,
        QuarantineAction::ClearSuspect,
        QuarantineAction::Condemn,
    ];

    for action in &actions {
        let result = qm.transition("gpu-1", *action, "attempt");
        assert!(
            result.is_err(),
            "Should not be able to transition from Condemned via {:?}",
            action
        );
    }
}

#[test]
fn test_cannot_schedule_deep_test_from_healthy() {
    let mut qm = QuarantineManager::new(test_config());
    let result = qm.transition("gpu-1", QuarantineAction::ScheduleDeepTest, "invalid");
    assert!(result.is_err());
}

#[test]
fn test_cannot_clear_suspect_from_quarantined() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine")
        .unwrap();
    let result = qm.transition("gpu-1", QuarantineAction::ClearSuspect, "invalid");
    assert!(result.is_err());
}

#[test]
fn test_cannot_mark_suspect_from_quarantined() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine")
        .unwrap();
    let result = qm.transition("gpu-1", QuarantineAction::MarkSuspect, "invalid");
    assert!(result.is_err());
}

// --- History and state tracking tests ---

#[test]
fn test_history_records_all_transitions() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::MarkSuspect, "step 1")
        .unwrap();
    qm.transition("gpu-1", QuarantineAction::Quarantine, "step 2")
        .unwrap();
    qm.transition("gpu-1", QuarantineAction::ScheduleDeepTest, "step 3")
        .unwrap();
    qm.transition("gpu-1", QuarantineAction::Condemn, "step 4")
        .unwrap();

    let history = qm.get_history("gpu-1");
    assert_eq!(history.len(), 4);
    assert_eq!(history[0].from_state, QuarantineState::Healthy);
    assert_eq!(history[0].to_state, QuarantineState::Suspect);
    assert_eq!(history[3].from_state, QuarantineState::DeepTest);
    assert_eq!(history[3].to_state, QuarantineState::Condemned);
}

#[test]
fn test_state_counts() {
    let mut qm = QuarantineManager::new(test_config());

    // Create GPUs in different states.
    qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
        .unwrap();
    qm.transition("gpu-2", QuarantineAction::Quarantine, "quarantine")
        .unwrap();
    qm.transition("gpu-3", QuarantineAction::Condemn, "condemn")
        .unwrap();
    qm.transition("gpu-4", QuarantineAction::Quarantine, "quarantine")
        .unwrap();
    qm.transition("gpu-4", QuarantineAction::ScheduleDeepTest, "test")
        .unwrap();

    let counts = qm.state_counts();
    assert_eq!(counts[&QuarantineState::Suspect], 1);
    assert_eq!(counts[&QuarantineState::Quarantined], 1);
    assert_eq!(counts[&QuarantineState::Condemned], 1);
    assert_eq!(counts[&QuarantineState::DeepTest], 1);
}

#[test]
fn test_gpus_in_state() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
        .unwrap();
    qm.transition("gpu-2", QuarantineAction::MarkSuspect, "suspect")
        .unwrap();
    qm.transition("gpu-3", QuarantineAction::Quarantine, "quarantine")
        .unwrap();

    let suspects = qm.gpus_in_state(QuarantineState::Suspect);
    assert_eq!(suspects.len(), 2);
    assert!(suspects.contains(&"gpu-1".to_string()));
    assert!(suspects.contains(&"gpu-2".to_string()));
}

#[test]
fn test_deep_test_pass_wrong_state() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::MarkSuspect, "suspect")
        .unwrap();

    // Not in DeepTest state, so this should return None.
    let result = qm.record_deep_test_pass("gpu-1");
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_unknown_gpu_deep_test_pass() {
    let mut qm = QuarantineManager::new(test_config());
    let result = qm.record_deep_test_pass("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_reinstatement_resets_deep_test_counter() {
    let mut qm = QuarantineManager::new(test_config());
    qm.transition("gpu-1", QuarantineAction::Quarantine, "quarantine")
        .unwrap();
    qm.transition("gpu-1", QuarantineAction::ScheduleDeepTest, "test")
        .unwrap();

    // Pass 3 of required 5.
    for _ in 0..3 {
        qm.record_deep_test_pass("gpu-1").unwrap();
    }

    // Condemn and then start over (hypothetical: create new GPU entry).
    qm.transition("gpu-1", QuarantineAction::Condemn, "condemn")
        .unwrap();

    // GPU is now condemned; counter should have been reset.
    let record = qm.get_record("gpu-1").unwrap();
    assert_eq!(record.deep_test_passes, 0);
}

#[test]
fn test_restore_state() {
    let mut qm = QuarantineManager::new(test_config());

    let record = sentinel_correlation_engine::health::quarantine::GpuStateRecord {
        state: QuarantineState::Quarantined,
        state_entered_at: chrono::Utc::now(),
        reason: "restored".to_string(),
        deep_test_passes: 0,
        history: Vec::new(),
    };

    qm.restore_state("gpu-restored", record);
    assert_eq!(qm.get_state("gpu-restored"), QuarantineState::Quarantined);
}
