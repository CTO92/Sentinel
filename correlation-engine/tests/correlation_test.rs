//! Integration tests for the full correlation pipeline.
//!
//! Tests the complete flow: event ingestion -> temporal window -> pattern
//! matching -> Bayesian update -> state transition.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tokio::sync::{mpsc, RwLock};

use sentinel_correlation_engine::correlation::engine::{
    CorrelationEngine, EngineCommand,
};
use sentinel_correlation_engine::correlation::temporal_window::{CorrelationEvent, EventType};
use sentinel_correlation_engine::health::quarantine::{QuarantineManager, QuarantineState};
use sentinel_correlation_engine::util::config::{AppConfig, QuarantineConfig};

fn make_event(gpu: &str, event_type: EventType, sm_id: Option<u32>) -> CorrelationEvent {
    CorrelationEvent {
        event_id: uuid::Uuid::new_v4().to_string(),
        gpu_uuid: gpu.to_string(),
        sm_id,
        event_type,
        timestamp: Utc::now(),
        severity: 2,
        score: 1.0,
        metadata: HashMap::new(),
    }
}

#[tokio::test]
async fn test_probe_pass_keeps_healthy() {
    let config = Arc::new(AppConfig::default());
    let qm = Arc::new(RwLock::new(QuarantineManager::new(
        config.quarantine.clone(),
    )));
    let (handle, _q_rx, cmd_rx) = CorrelationEngine::new(config.clone(), qm.clone());

    let (q_tx, _) = mpsc::channel(100);
    let engine_task = tokio::spawn(async move {
        CorrelationEngine::run(config, qm.clone(), q_tx, cmd_rx).await;
    });

    // Send 100 probe passes.
    for _ in 0..100 {
        handle
            .process_event(make_event("gpu-1", EventType::ProbePass, Some(0)))
            .await
            .unwrap();
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    let snapshot = handle
        .query_gpu_health("gpu-1".to_string())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(snapshot.probe_pass_count, 100);
    assert!(snapshot.reliability_score > 0.999);

    handle.shutdown().await.unwrap();
    let _ = engine_task.await;
}

#[tokio::test]
async fn test_probe_failures_trigger_state_transitions() {
    let config = Arc::new(AppConfig::default());
    let qm = Arc::new(RwLock::new(QuarantineManager::new(
        config.quarantine.clone(),
    )));
    let (handle, mut q_rx, cmd_rx) = CorrelationEngine::new(config.clone(), qm.clone());

    let (q_tx, _) = mpsc::channel(100);
    let engine_task = tokio::spawn(async move {
        CorrelationEngine::run(config, qm.clone(), q_tx, cmd_rx).await;
    });

    // Send enough probe failures to push below suspect threshold (0.995).
    // reliability = 1000 / (1000 + 1 + n) < 0.995
    // => n > 1000/0.995 - 1001 = 1005.025... - 1001 = 4.025
    // So 6 failures should push below 0.995.
    for _ in 0..6 {
        handle
            .process_event(make_event("gpu-1", EventType::ProbeFail, Some(0)))
            .await
            .unwrap();
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    let snapshot = handle
        .query_gpu_health("gpu-1".to_string())
        .await
        .unwrap()
        .unwrap();
    assert!(snapshot.reliability_score < 0.995);
    assert!(snapshot.probe_fail_count >= 6);

    handle.shutdown().await.unwrap();
    let _ = engine_task.await;
}

#[tokio::test]
async fn test_mixed_events_correlation() {
    let config = Arc::new(AppConfig::default());
    let qm = Arc::new(RwLock::new(QuarantineManager::new(
        config.quarantine.clone(),
    )));
    let (handle, _q_rx, cmd_rx) = CorrelationEngine::new(config.clone(), qm.clone());

    let (q_tx, _) = mpsc::channel(100);
    let engine_task = tokio::spawn(async move {
        CorrelationEngine::run(config, qm.clone(), q_tx, cmd_rx).await;
    });

    // Send a mix of probes and anomalies.
    handle
        .process_event(make_event("gpu-1", EventType::ProbeFail, Some(3)))
        .await
        .unwrap();
    handle
        .process_event(make_event("gpu-1", EventType::InferenceAnomaly, None))
        .await
        .unwrap();
    handle
        .process_event(make_event("gpu-1", EventType::ThermalSpike, None))
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    let snapshot = handle
        .query_gpu_health("gpu-1".to_string())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(snapshot.probe_fail_count, 1);
    assert_eq!(snapshot.anomaly_count, 1);

    handle.shutdown().await.unwrap();
    let _ = engine_task.await;
}

#[tokio::test]
async fn test_fleet_health_query() {
    let config = Arc::new(AppConfig::default());
    let qm = Arc::new(RwLock::new(QuarantineManager::new(
        config.quarantine.clone(),
    )));
    let (handle, _q_rx, cmd_rx) = CorrelationEngine::new(config.clone(), qm.clone());

    let (q_tx, _) = mpsc::channel(100);
    let engine_task = tokio::spawn(async move {
        CorrelationEngine::run(config, qm.clone(), q_tx, cmd_rx).await;
    });

    // Register multiple GPUs by sending events.
    for i in 1..=5 {
        handle
            .process_event(make_event(
                &format!("gpu-{}", i),
                EventType::ProbePass,
                None,
            ))
            .await
            .unwrap();
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    let fleet = handle.query_fleet_health().await.unwrap();
    assert_eq!(fleet.total_gpus, 5);
    assert!(fleet.average_reliability > 0.99);

    handle.shutdown().await.unwrap();
    let _ = engine_task.await;
}

#[tokio::test]
async fn test_gpu_history_query() {
    let config = Arc::new(AppConfig::default());
    let qm = Arc::new(RwLock::new(QuarantineManager::new(
        config.quarantine.clone(),
    )));
    let (handle, _q_rx, cmd_rx) = CorrelationEngine::new(config.clone(), qm.clone());

    let (q_tx, _) = mpsc::channel(100);
    let engine_task = tokio::spawn(async move {
        CorrelationEngine::run(config, qm.clone(), q_tx, cmd_rx).await;
    });

    for _ in 0..10 {
        handle
            .process_event(make_event("gpu-1", EventType::ProbePass, None))
            .await
            .unwrap();
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    let history = handle
        .query_gpu_history("gpu-1".to_string(), 5)
        .await
        .unwrap();
    assert_eq!(history.len(), 5);

    handle.shutdown().await.unwrap();
    let _ = engine_task.await;
}
