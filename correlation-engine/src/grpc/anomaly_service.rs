//! gRPC AnomalyService implementation.
//!
//! Handles bidirectional streaming of anomaly events from inference and
//! training monitors. Converts protobuf anomaly events into internal
//! correlation events and forwards them to the correlation engine.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tonic::Streaming;
use tracing::{debug, error, info, warn};

use crate::correlation::engine::EngineHandle;
use crate::correlation::temporal_window::{CorrelationEvent, EventType};
use crate::util::metrics;

/// Implementation of the AnomalyService gRPC service.
pub struct AnomalyServiceImpl {
    /// Handle to the correlation engine.
    engine_handle: EngineHandle,
}

impl AnomalyServiceImpl {
    /// Create a new AnomalyService implementation.
    pub fn new(engine_handle: EngineHandle) -> Self {
        Self { engine_handle }
    }

    /// Process a single anomaly event, converting it to an internal
    /// correlation event and forwarding to the engine.
    #[allow(clippy::too_many_arguments)]
    pub async fn process_anomaly_event(
        &self,
        event_id: &str,
        anomaly_type: i32,
        source: i32,
        gpu_uuid: &str,
        severity: u32,
        score: f64,
        hostname: &str,
        timestamp: Option<DateTime<Utc>>,
        layer_name: Option<&str>,
        model_id: Option<&str>,
    ) {
        let event_type = match source {
            1 => EventType::InferenceAnomaly,   // ANOMALY_SOURCE_INFERENCE_MONITOR
            2 => EventType::TrainingAnomaly,    // ANOMALY_SOURCE_TRAINING_MONITOR
            3 => EventType::InvariantViolation, // ANOMALY_SOURCE_INVARIANT_CHECKER
            _ => EventType::InferenceAnomaly,
        };

        let mut metadata = HashMap::new();
        metadata.insert("hostname".to_string(), hostname.to_string());
        metadata.insert("anomaly_type".to_string(), anomaly_type.to_string());
        if let Some(layer) = layer_name {
            metadata.insert("layer_name".to_string(), layer.to_string());
        }
        if let Some(model) = model_id {
            metadata.insert("model_id".to_string(), model.to_string());
        }

        let event = CorrelationEvent {
            event_id: event_id.to_string(),
            gpu_uuid: gpu_uuid.to_string(),
            sm_id: None,
            event_type,
            timestamp: timestamp.unwrap_or_else(Utc::now),
            severity,
            score,
            metadata,
        };

        metrics::record_event_ingested("anomaly");

        if let Err(e) = self.engine_handle.process_event(event).await {
            error!(
                event_id = %event_id,
                error = %e,
                "Failed to forward anomaly event to engine"
            );
        }
    }

    /// Handle the StreamAnomalyEvents bidirectional stream.
    pub async fn handle_stream(
        &self,
        mut inbound: Streaming<serde_json::Value>,
        ack_tx: mpsc::Sender<serde_json::Value>,
    ) {
        info!("Anomaly monitor stream connected");

        while let Some(result) = inbound.next().await {
            match result {
                Ok(batch) => {
                    let seq = batch
                        .get("sequence_number")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let hostname = batch
                        .get("source_hostname")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");

                    debug!(
                        hostname = %hostname,
                        sequence = seq,
                        "Received anomaly event batch"
                    );

                    let ack = serde_json::json!({
                        "sequence_number": seq,
                        "accepted": true,
                    });

                    if let Err(e) = ack_tx.send(ack).await {
                        warn!(error = %e, "Failed to send anomaly ack");
                        break;
                    }
                }
                Err(e) => {
                    error!(error = %e, "Error receiving anomaly batch");
                    break;
                }
            }
        }

        info!("Anomaly monitor stream disconnected");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlation::engine::CorrelationEngine;
    use crate::health::quarantine::QuarantineManager;
    use crate::util::config::{AppConfig, QuarantineConfig};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[tokio::test]
    async fn test_process_anomaly() {
        let config = Arc::new(AppConfig::default());
        let qm = Arc::new(RwLock::new(QuarantineManager::new(
            QuarantineConfig::default(),
        )));
        let (handle, _q_rx, cmd_rx) = CorrelationEngine::new(config.clone(), qm.clone());

        let service = AnomalyServiceImpl::new(handle.clone());

        let (q_tx, _) = mpsc::channel(100);
        let engine_task = tokio::spawn(async move {
            CorrelationEngine::run(config, qm, q_tx, cmd_rx).await;
        });

        service
            .process_anomaly_event(
                "anomaly-1",
                1,  // LOGIT_DRIFT
                1,  // INFERENCE_MONITOR
                "gpu-1",
                3,
                0.85,
                "host-1",
                None,
                Some("layer.0.attention"),
                Some("llama-70b"),
            )
            .await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        handle.shutdown().await.unwrap();
        let _ = engine_task.await;
    }
}
