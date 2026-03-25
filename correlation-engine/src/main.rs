//! SENTINEL Correlation Engine entry point.
//!
//! Starts all subsystems:
//! - gRPC server for probe/anomaly/query/config services
//! - Correlation engine actor
//! - TMR scheduler (periodic task)
//! - Alert dispatcher
//! - REST API server (axum, for dashboard)
//! - Prometheus metrics server
//!
//! Uses Tokio multi-threaded runtime for concurrent task execution.

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{mpsc, RwLock};
use tracing::{error, info};

use sentinel_correlation_engine::alerting::alert_manager::AlertManager;
use sentinel_correlation_engine::correlation::engine::CorrelationEngine;
use sentinel_correlation_engine::grpc::server::{self, AppState};
use sentinel_correlation_engine::health::quarantine::QuarantineManager;
use sentinel_correlation_engine::storage::audit_client::AuditClient;
use sentinel_correlation_engine::storage::event_store::{EventStore, InMemoryPersistence};
use sentinel_correlation_engine::storage::state_store::StateStore;
use sentinel_correlation_engine::trust::tmr_scheduler::TmrScheduler;
use sentinel_correlation_engine::trust::trust_graph::TrustGraph;
use sentinel_correlation_engine::util::config::{load_config, AppConfig};
use sentinel_correlation_engine::util::metrics;

#[tokio::main]
async fn main() -> Result<()> {
    // Load configuration.
    let config = Arc::new(load_config(None).unwrap_or_else(|e| {
        eprintln!("Warning: failed to load config ({e}), using defaults");
        AppConfig::default()
    }));

    // Initialize logging.
    init_logging(&config);
    info!("SENTINEL Correlation Engine starting");

    // Initialize metrics.
    metrics::init_metrics();

    // Initialize state store (Redis or in-memory fallback).
    let state_store = Arc::new(StateStore::new(&config.redis).await);
    if state_store.is_redis_connected() {
        info!("State store: Redis");
    } else {
        info!("State store: in-memory (Redis unavailable)");
    }

    // Initialize audit client.
    let audit_client = Arc::new(AuditClient::new(config.audit.clone()));
    audit_client
        .record_system_event("Correlation engine starting")
        .await?;

    // Initialize quarantine manager and restore state.
    let quarantine_manager = Arc::new(RwLock::new(QuarantineManager::new(
        config.quarantine.clone(),
    )));

    // Restore GPU states from state store.
    match state_store.load_all_gpu_states().await {
        Ok(states) => {
            let mut qm = quarantine_manager.write().await;
            for (uuid, record) in states {
                info!(gpu = %uuid, state = ?record.state, "Restored GPU state");
                qm.restore_state(&uuid, record);
            }
        }
        Err(e) => {
            error!(error = %e, "Failed to restore GPU states");
        }
    }

    // Initialize trust graph.
    let trust_graph = Arc::new(RwLock::new(TrustGraph::new(config.trust.clone())));

    // Initialize the correlation engine.
    let (engine_handle, mut quarantine_rx, cmd_rx) =
        CorrelationEngine::new(config.clone(), quarantine_manager.clone());

    // Initialize alert manager.
    let (alert_handle, alert_rx) = AlertManager::new(config.alerting.clone());

    // Initialize event store.
    let persistence = Arc::new(InMemoryPersistence::new(config.event_store.ring_buffer_size));
    let (event_store, event_store_handle) =
        EventStore::new(config.event_store.clone(), persistence);

    // Initialize TMR scheduler.
    let tmr_scheduler = Arc::new(TmrScheduler::new(
        config.tmr.clone(),
        trust_graph.clone(),
        quarantine_manager.clone(),
    ));

    // Shared application state for REST API.
    let app_state = AppState {
        engine_handle: engine_handle.clone(),
        trust_graph: trust_graph.clone(),
        quarantine_manager: quarantine_manager.clone(),
    };

    // Spawn the correlation engine actor.
    let engine_config = config.clone();
    let engine_qm = quarantine_manager.clone();
    let (quarantine_action_tx, _) = mpsc::channel(1_000);
    let engine_task = tokio::spawn(async move {
        CorrelationEngine::run(engine_config, engine_qm, quarantine_action_tx, cmd_rx).await;
    });

    // Spawn the alert manager.
    let alert_config = config.alerting.clone();
    let alert_task = tokio::spawn(async move {
        AlertManager::run(alert_config, alert_rx).await;
    });

    // Spawn the audit client delivery loop.
    let audit_task = tokio::spawn(audit_client.clone().run_delivery_loop());

    // Spawn the TMR scheduler periodic task.
    let tmr_interval = config.tmr.interval_secs;
    let tmr_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(tmr_interval));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            let requests = tmr_scheduler.schedule_canaries().await;
            if !requests.is_empty() {
                info!(count = requests.len(), "TMR canaries scheduled");
            }
        }
    });

    // Spawn quarantine action processor.
    let quarantine_audit = audit_client.clone();
    let quarantine_state_store = state_store.clone();
    let quarantine_qm = quarantine_manager.clone();
    let quarantine_task = tokio::spawn(async move {
        while let Some((action, gpu_uuid, evidence)) = quarantine_rx.recv().await {
            // Record in audit trail.
            if let Err(e) = quarantine_audit
                .record_quarantine_action(
                    &gpu_uuid,
                    &format!("{}", action),
                    "Bayesian threshold triggered",
                    &evidence,
                )
                .await
            {
                error!(error = %e, "Failed to record quarantine action in audit");
            }

            // Persist state.
            let qm = quarantine_qm.read().await;
            if let Some(record) = qm.get_record(&gpu_uuid) {
                if let Err(e) = quarantine_state_store
                    .save_gpu_state(&gpu_uuid, record)
                    .await
                {
                    error!(error = %e, "Failed to persist GPU state");
                }
            }

            info!(
                gpu = %gpu_uuid,
                action = %action,
                "Quarantine action processed"
            );
        }
    });

    // Spawn REST API server.
    let rest_config = config.clone();
    let rest_task = tokio::spawn(async move {
        if let Err(e) = server::start_rest_server(&rest_config, app_state).await {
            error!(error = %e, "REST API server failed");
        }
    });

    // Spawn metrics server.
    let metrics_config = config.clone();
    let metrics_task = tokio::spawn(async move {
        if let Err(e) = server::start_metrics_server(&metrics_config).await {
            error!(error = %e, "Metrics server failed");
        }
    });

    info!("SENTINEL Correlation Engine fully started");

    // Wait for shutdown signal.
    tokio::signal::ctrl_c().await?;
    info!("Shutdown signal received");

    // Graceful shutdown.
    engine_handle.shutdown().await?;
    audit_client
        .record_system_event("Correlation engine shutting down")
        .await?;

    info!("SENTINEL Correlation Engine stopped");
    Ok(())
}

/// Initialize the tracing subscriber for structured logging.
fn init_logging(config: &AppConfig) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&config.logging.level));

    if config.logging.json {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .json()
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .init();
    }
}
