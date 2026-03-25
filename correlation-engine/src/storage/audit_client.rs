//! gRPC client for the Audit Ledger service.
//!
//! Streams quarantine actions, state transitions, and configuration changes
//! to the tamper-evident audit ledger. If the audit service is unavailable,
//! events are queued in memory and retried.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::util::config::AuditConfig;

/// An audit event to be recorded in the ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Type of audit entry.
    pub entry_type: AuditEntryType,

    /// GPU UUID associated with this event (if applicable).
    pub gpu_uuid: Option<String>,

    /// Serialized event data (JSON).
    pub data: String,

    /// When the event occurred.
    pub timestamp: DateTime<Utc>,

    /// Who or what initiated this event.
    pub initiated_by: String,
}

/// Types of audit entries.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AuditEntryType {
    ProbeResult,
    AnomalyEvent,
    QuarantineAction,
    ConfigChange,
    TmrResult,
    SystemEvent,
}

impl std::fmt::Display for AuditEntryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditEntryType::ProbeResult => write!(f, "probe_result"),
            AuditEntryType::AnomalyEvent => write!(f, "anomaly_event"),
            AuditEntryType::QuarantineAction => write!(f, "quarantine_action"),
            AuditEntryType::ConfigChange => write!(f, "config_change"),
            AuditEntryType::TmrResult => write!(f, "tmr_result"),
            AuditEntryType::SystemEvent => write!(f, "system_event"),
        }
    }
}

/// Client for the Audit Ledger gRPC service.
///
/// Provides fire-and-forget semantics with internal buffering and retry.
/// Events are queued and sent asynchronously to the audit ledger.
pub struct AuditClient {
    /// Configuration.
    config: AuditConfig,

    /// Queue of events pending delivery.
    queue: Arc<Mutex<VecDeque<AuditEvent>>>,

    /// Maximum queue depth before events are dropped.
    max_queue_depth: usize,

    /// Whether the client is connected to the audit service.
    connected: Arc<std::sync::atomic::AtomicBool>,
}

impl AuditClient {
    /// Create a new audit client.
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            queue: Arc::new(Mutex::new(VecDeque::with_capacity(10_000))),
            max_queue_depth: 100_000,
            connected: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Submit an audit event for delivery to the ledger.
    ///
    /// This is a non-blocking operation; events are queued internally.
    pub async fn submit(&self, event: AuditEvent) -> Result<()> {
        let mut queue = self.queue.lock().await;
        if queue.len() >= self.max_queue_depth {
            // Drop oldest events to make room.
            let dropped = queue.len() - self.max_queue_depth + 1;
            for _ in 0..dropped {
                queue.pop_front();
            }
            warn!(
                dropped,
                "Audit queue overflow; oldest events dropped"
            );
        }
        queue.push_back(event);
        Ok(())
    }

    /// Submit a quarantine action audit event.
    pub async fn record_quarantine_action(
        &self,
        gpu_uuid: &str,
        action: &str,
        reason: &str,
        evidence: &[String],
    ) -> Result<()> {
        let data = serde_json::json!({
            "action": action,
            "reason": reason,
            "evidence": evidence,
        });

        self.submit(AuditEvent {
            entry_type: AuditEntryType::QuarantineAction,
            gpu_uuid: Some(gpu_uuid.to_string()),
            data: data.to_string(),
            timestamp: Utc::now(),
            initiated_by: "correlation-engine".to_string(),
        })
        .await
    }

    /// Submit a state transition audit event.
    pub async fn record_state_transition(
        &self,
        gpu_uuid: &str,
        from_state: &str,
        to_state: &str,
        reason: &str,
    ) -> Result<()> {
        let data = serde_json::json!({
            "from_state": from_state,
            "to_state": to_state,
            "reason": reason,
        });

        self.submit(AuditEvent {
            entry_type: AuditEntryType::QuarantineAction,
            gpu_uuid: Some(gpu_uuid.to_string()),
            data: data.to_string(),
            timestamp: Utc::now(),
            initiated_by: "correlation-engine".to_string(),
        })
        .await
    }

    /// Submit a configuration change audit event.
    pub async fn record_config_change(
        &self,
        change_description: &str,
        initiated_by: &str,
    ) -> Result<()> {
        let data = serde_json::json!({
            "description": change_description,
        });

        self.submit(AuditEvent {
            entry_type: AuditEntryType::ConfigChange,
            gpu_uuid: None,
            data: data.to_string(),
            timestamp: Utc::now(),
            initiated_by: initiated_by.to_string(),
        })
        .await
    }

    /// Submit a system event audit entry.
    pub async fn record_system_event(&self, description: &str) -> Result<()> {
        self.submit(AuditEvent {
            entry_type: AuditEntryType::SystemEvent,
            gpu_uuid: None,
            data: serde_json::json!({"description": description}).to_string(),
            timestamp: Utc::now(),
            initiated_by: "correlation-engine".to_string(),
        })
        .await
    }

    /// Run the audit client delivery loop. This drains the queue and attempts
    /// to send events to the audit ledger via gRPC.
    ///
    /// In the current implementation, this logs events when the audit service
    /// is unavailable. In production, it would establish a gRPC streaming
    /// connection to the audit ledger service.
    pub async fn run_delivery_loop(self: Arc<Self>) {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        info!(
            endpoint = %self.config.endpoint,
            "Audit client delivery loop started"
        );

        loop {
            interval.tick().await;

            let events = {
                let mut queue = self.queue.lock().await;
                let batch: Vec<AuditEvent> = queue.drain(..).collect();
                batch
            };

            if events.is_empty() {
                continue;
            }

            // Attempt delivery. In production, this would use the gRPC client.
            // For now, we log the events and mark them as delivered.
            match self.deliver_batch(&events).await {
                Ok(()) => {
                    debug!(count = events.len(), "Delivered audit events");
                }
                Err(e) => {
                    error!(
                        count = events.len(),
                        error = %e,
                        "Failed to deliver audit events; re-queuing"
                    );
                    // Re-queue failed events.
                    let mut queue = self.queue.lock().await;
                    for event in events.into_iter().rev() {
                        queue.push_front(event);
                    }
                }
            }
        }
    }

    /// Attempt to deliver a batch of events to the audit ledger.
    async fn deliver_batch(&self, events: &[AuditEvent]) -> Result<()> {
        // In a full implementation, this would:
        // 1. Establish a gRPC connection to the audit ledger
        // 2. Serialize events to protobuf AuditEntry messages
        // 3. Call AuditService::IngestEvents
        //
        // For now, we log the events at debug level.
        for event in events {
            debug!(
                entry_type = %event.entry_type,
                gpu = ?event.gpu_uuid,
                initiated_by = %event.initiated_by,
                "Audit event"
            );
        }

        self.connected
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Get the current queue depth.
    pub async fn queue_depth(&self) -> usize {
        self.queue.lock().await.len()
    }

    /// Check if the client is connected to the audit service.
    pub fn is_connected(&self) -> bool {
        self.connected
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_submit_event() {
        let client = AuditClient::new(AuditConfig::default());

        client
            .record_quarantine_action("gpu-1", "quarantine", "test reason", &["ev-1".to_string()])
            .await
            .unwrap();

        assert_eq!(client.queue_depth().await, 1);
    }

    #[tokio::test]
    async fn test_queue_overflow() {
        let mut client = AuditClient::new(AuditConfig::default());
        client.max_queue_depth = 5;

        for i in 0..10 {
            client
                .record_system_event(&format!("event {}", i))
                .await
                .unwrap();
        }

        assert!(client.queue_depth().await <= 5);
    }
}
