//! gRPC CorrelationService (query) implementation.
//!
//! Provides query APIs for the dashboard and external consumers:
//! - QueryGpuHealth: Get health state for a specific GPU
//! - QueryFleetHealth: Get fleet-wide summary
//! - GetGpuHistory: Get event history for a GPU

use std::sync::Arc;

use tokio::sync::RwLock;
use crate::correlation::engine::EngineHandle;
use crate::health::quarantine::QuarantineManager;

/// Implementation of the CorrelationService (query) gRPC service.
pub struct QueryServiceImpl {
    /// Handle to the correlation engine for queries.
    engine_handle: EngineHandle,

    /// Direct access to quarantine manager for state queries.
    quarantine_manager: Arc<RwLock<QuarantineManager>>,
}

impl QueryServiceImpl {
    /// Create a new QueryService implementation.
    pub fn new(
        engine_handle: EngineHandle,
        quarantine_manager: Arc<RwLock<QuarantineManager>>,
    ) -> Self {
        Self {
            engine_handle,
            quarantine_manager,
        }
    }

    /// Query the health of a specific GPU.
    ///
    /// Returns the current health state including Bayesian reliability score,
    /// quarantine state, probe/anomaly counts, and recent correlation patterns.
    pub async fn query_gpu_health(
        &self,
        gpu_uuid: &str,
    ) -> Result<Option<GpuHealthResponse>, anyhow::Error> {
        let snapshot = self
            .engine_handle
            .query_gpu_health(gpu_uuid.to_string())
            .await?;

        match snapshot {
            Some(snap) => Ok(Some(GpuHealthResponse {
                gpu_uuid: snap.gpu_uuid,
                state: format!("{}", snap.state),
                reliability_score: snap.reliability_score,
                alpha: snap.alpha,
                beta: snap.beta,
                probe_pass_count: snap.probe_pass_count,
                probe_fail_count: snap.probe_fail_count,
                anomaly_count: snap.anomaly_count,
                recent_pattern_count: snap.recent_patterns.len() as u32,
            })),
            None => Ok(None),
        }
    }

    /// Query fleet-wide health summary.
    pub async fn query_fleet_health(&self) -> Result<FleetHealthResponse, anyhow::Error> {
        let snapshot = self.engine_handle.query_fleet_health().await?;

        Ok(FleetHealthResponse {
            total_gpus: snapshot.total_gpus,
            healthy: snapshot.healthy,
            suspect: snapshot.suspect,
            quarantined: snapshot.quarantined,
            deep_test: snapshot.deep_test,
            condemned: snapshot.condemned,
            average_reliability: snapshot.average_reliability,
        })
    }

    /// Query event history for a GPU.
    pub async fn get_gpu_history(
        &self,
        gpu_uuid: &str,
        limit: usize,
    ) -> Result<Vec<EventSummary>, anyhow::Error> {
        let events = self
            .engine_handle
            .query_gpu_history(gpu_uuid.to_string(), limit)
            .await?;

        Ok(events
            .into_iter()
            .map(|e| EventSummary {
                event_id: e.event_id,
                event_type: e.event_type.to_string(),
                timestamp: e.timestamp.to_rfc3339(),
                severity: e.severity,
                score: e.score,
            })
            .collect())
    }
}

/// Response structure for GPU health queries.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuHealthResponse {
    pub gpu_uuid: String,
    pub state: String,
    pub reliability_score: f64,
    pub alpha: f64,
    pub beta: f64,
    pub probe_pass_count: u64,
    pub probe_fail_count: u64,
    pub anomaly_count: u64,
    pub recent_pattern_count: u32,
}

/// Response structure for fleet health queries.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FleetHealthResponse {
    pub total_gpus: u32,
    pub healthy: u32,
    pub suspect: u32,
    pub quarantined: u32,
    pub deep_test: u32,
    pub condemned: u32,
    pub average_reliability: f64,
}

/// Summary of a single event for history queries.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EventSummary {
    pub event_id: String,
    pub event_type: String,
    pub timestamp: String,
    pub severity: u32,
    pub score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlation::engine::CorrelationEngine;
    use crate::util::config::{AppConfig, QuarantineConfig};
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_query_unknown_gpu() {
        let config = Arc::new(AppConfig::default());
        let qm = Arc::new(RwLock::new(QuarantineManager::new(
            QuarantineConfig::default(),
        )));
        let (handle, _q_rx, cmd_rx) = CorrelationEngine::new(config.clone(), qm.clone());

        let (q_tx, _) = mpsc::channel(100);
        let engine_task = tokio::spawn(async move {
            CorrelationEngine::run(config, qm.clone(), q_tx, cmd_rx).await;
        });

        let service = QueryServiceImpl::new(
            handle.clone(),
            Arc::new(RwLock::new(QuarantineManager::new(
                QuarantineConfig::default(),
            ))),
        );

        let result = service.query_gpu_health("nonexistent-gpu").await.unwrap();
        assert!(result.is_none());

        handle.shutdown().await.unwrap();
        let _ = engine_task.await;
    }
}
