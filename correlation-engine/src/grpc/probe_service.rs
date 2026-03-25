//! gRPC ProbeService implementation.
//!
//! Handles bidirectional streaming of probe results from probe agents.
//! Converts protobuf probe results into internal correlation events and
//! forwards them to the correlation engine.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tonic::Streaming;
use tracing::{debug, error, info, warn};

use crate::correlation::engine::EngineHandle;
use crate::correlation::temporal_window::{CorrelationEvent, EventType};
use crate::util::metrics;

/// Implementation of the ProbeService gRPC service.
///
/// This service receives streaming probe results from probe agents and
/// forwards them to the correlation engine for processing.
pub struct ProbeServiceImpl {
    /// Handle to the correlation engine.
    engine_handle: EngineHandle,
}

impl ProbeServiceImpl {
    /// Create a new ProbeService implementation.
    pub fn new(engine_handle: EngineHandle) -> Self {
        Self { engine_handle }
    }

    /// Process a single probe execution from a batch, converting it to
    /// an internal correlation event and forwarding to the engine.
    async fn process_probe_execution(
        &self,
        agent_hostname: &str,
        execution_id: &str,
        probe_type: i32,
        result_code: i32,
        gpu_uuid: &str,
        sm_id: Option<u32>,
        temperature: f32,
        power: f32,
        timestamp: Option<DateTime<Utc>>,
    ) {
        let event_type = match result_code {
            1 => EventType::ProbePass,   // PROBE_RESULT_PASS
            2 => EventType::ProbeFail,   // PROBE_RESULT_FAIL
            3 => EventType::ProbeError,  // PROBE_RESULT_ERROR
            4 => EventType::ProbeError,  // PROBE_RESULT_TIMEOUT
            _ => EventType::ProbeError,
        };

        let severity = match result_code {
            1 => 1, // INFO for pass
            2 => 4, // CRITICAL for fail
            3 => 3, // HIGH for error
            4 => 2, // WARNING for timeout
            _ => 2,
        };

        let mut metadata = HashMap::new();
        metadata.insert("hostname".to_string(), agent_hostname.to_string());
        metadata.insert("probe_type".to_string(), probe_type.to_string());
        metadata.insert("temperature_c".to_string(), temperature.to_string());
        metadata.insert("power_w".to_string(), power.to_string());

        let event = CorrelationEvent {
            event_id: execution_id.to_string(),
            gpu_uuid: gpu_uuid.to_string(),
            sm_id,
            event_type,
            timestamp: timestamp.unwrap_or_else(Utc::now),
            severity,
            score: if result_code == 2 { 1.0 } else { 0.0 },
            metadata,
        };

        // Record metric.
        metrics::record_event_ingested("probe");

        if let Err(e) = self.engine_handle.process_event(event).await {
            error!(
                execution_id = %execution_id,
                error = %e,
                "Failed to forward probe event to engine"
            );
        }
    }
}

/// Convert a prost Timestamp to chrono DateTime.
fn prost_timestamp_to_chrono(ts: &prost_types::Timestamp) -> DateTime<Utc> {
    DateTime::from_timestamp(ts.seconds, ts.nanos as u32).unwrap_or_else(Utc::now)
}

// Note: The actual tonic service trait implementation requires the generated
// protobuf types. Below is the service logic that would be called from the
// generated trait implementation.

impl ProbeServiceImpl {
    /// Handle the StreamProbeResults bidirectional stream.
    ///
    /// This method processes incoming probe result batches and sends back
    /// acknowledgments. In the actual implementation, this would be called
    /// from the generated `ProbeService` trait.
    pub async fn handle_stream(
        &self,
        mut inbound: Streaming<serde_json::Value>,
        ack_tx: mpsc::Sender<serde_json::Value>,
    ) {
        info!("Probe agent stream connected");

        while let Some(result) = inbound.next().await {
            match result {
                Ok(batch) => {
                    let seq = batch
                        .get("sequence_number")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let hostname = batch
                        .get("agent_hostname")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");

                    debug!(
                        hostname = %hostname,
                        sequence = seq,
                        "Received probe result batch"
                    );

                    // Send ack.
                    let ack = serde_json::json!({
                        "sequence_number": seq,
                        "accepted": true,
                    });

                    if let Err(e) = ack_tx.send(ack).await {
                        warn!(error = %e, "Failed to send probe ack");
                        break;
                    }
                }
                Err(e) => {
                    error!(error = %e, "Error receiving probe batch");
                    break;
                }
            }
        }

        info!("Probe agent stream disconnected");
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
    async fn test_process_probe_pass() {
        let config = Arc::new(AppConfig::default());
        let qm = Arc::new(RwLock::new(QuarantineManager::new(
            QuarantineConfig::default(),
        )));
        let (handle, _q_rx, cmd_rx) = CorrelationEngine::new(config.clone(), qm.clone());

        let service = ProbeServiceImpl::new(handle.clone());

        let (q_tx, _) = mpsc::channel(100);
        let engine_task = tokio::spawn(async move {
            CorrelationEngine::run(config, qm, q_tx, cmd_rx).await;
        });

        service
            .process_probe_execution(
                "host-1",
                "exec-1",
                1,  // FMA
                1,  // PASS
                "gpu-1",
                Some(0),
                72.0,
                350.0,
                None,
            )
            .await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        handle.shutdown().await.unwrap();
        let _ = engine_task.await;
    }
}
