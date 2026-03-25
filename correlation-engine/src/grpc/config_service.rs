//! gRPC ConfigService implementation.
//!
//! Manages dynamic configuration updates pushed to probe agents and
//! monitoring components. Supports bidirectional streaming where the
//! engine pushes config updates and components send back acknowledgments.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, info, warn};

use crate::storage::audit_client::AuditClient;

/// A configuration update to be pushed to components.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConfigUpdateMessage {
    /// Unique update identifier.
    pub update_id: String,

    /// Who initiated the change.
    pub initiated_by: String,

    /// Reason for the change.
    pub reason: String,

    /// The configuration change (serialized as JSON).
    pub update_payload: serde_json::Value,

    /// Timestamp of the update.
    pub timestamp: chrono::DateTime<Utc>,
}

/// Acknowledgment from a component that received a config update.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConfigAckMessage {
    /// The update_id being acknowledged.
    pub update_id: String,

    /// Whether the update was applied successfully.
    pub applied: bool,

    /// Component that processed the update.
    pub component_id: String,

    /// Error message if not applied.
    pub error: Option<String>,

    /// New effective config version.
    pub config_version: u64,
}

/// The ConfigService manages dynamic configuration distribution.
pub struct ConfigServiceImpl {
    /// Broadcast channel for pushing updates to all connected components.
    update_tx: broadcast::Sender<ConfigUpdateMessage>,

    /// Pending updates awaiting acknowledgment.
    pending_updates: Arc<RwLock<HashMap<String, PendingUpdate>>>,

    /// Audit client for recording config changes.
    audit_client: Arc<AuditClient>,
}

/// Tracks a pending configuration update.
#[derive(Debug, Clone)]
struct PendingUpdate {
    update: ConfigUpdateMessage,
    acks_received: Vec<ConfigAckMessage>,
    expected_acks: usize,
}

impl ConfigServiceImpl {
    /// Create a new ConfigService implementation.
    pub fn new(audit_client: Arc<AuditClient>) -> Self {
        let (update_tx, _) = broadcast::channel(100);

        Self {
            update_tx,
            pending_updates: Arc::new(RwLock::new(HashMap::new())),
            audit_client,
        }
    }

    /// Push a configuration update to all connected components.
    pub async fn push_update(&self, update: ConfigUpdateMessage) -> anyhow::Result<()> {
        // Record in audit trail.
        self.audit_client
            .record_config_change(
                &serde_json::to_string(&update.update_payload)?,
                &update.initiated_by,
            )
            .await?;

        // Track as pending.
        {
            let mut pending = self.pending_updates.write().await;
            pending.insert(
                update.update_id.clone(),
                PendingUpdate {
                    update: update.clone(),
                    acks_received: Vec::new(),
                    expected_acks: 0, // Set based on connected clients.
                },
            );
        }

        // Broadcast to all connected components.
        let receivers = self.update_tx.receiver_count();
        if receivers > 0 {
            match self.update_tx.send(update.clone()) {
                Ok(sent) => {
                    info!(
                        update_id = %update.update_id,
                        receivers = sent,
                        "Config update broadcast"
                    );
                }
                Err(e) => {
                    warn!(
                        update_id = %update.update_id,
                        error = %e,
                        "No receivers for config update"
                    );
                }
            }
        } else {
            debug!(
                update_id = %update.update_id,
                "No connected components to receive config update"
            );
        }

        Ok(())
    }

    /// Process a configuration acknowledgment from a component.
    pub async fn process_ack(&self, ack: ConfigAckMessage) {
        let mut pending = self.pending_updates.write().await;

        if let Some(entry) = pending.get_mut(&ack.update_id) {
            entry.acks_received.push(ack.clone());

            if ack.applied {
                debug!(
                    update_id = %ack.update_id,
                    component = %ack.component_id,
                    "Config update acknowledged"
                );
            } else {
                warn!(
                    update_id = %ack.update_id,
                    component = %ack.component_id,
                    error = ?ack.error,
                    "Config update rejected by component"
                );
            }
        } else {
            warn!(
                update_id = %ack.update_id,
                "Received ack for unknown config update"
            );
        }
    }

    /// Subscribe to configuration updates.
    pub fn subscribe(&self) -> broadcast::Receiver<ConfigUpdateMessage> {
        self.update_tx.subscribe()
    }

    /// Get the number of pending (unacknowledged) updates.
    pub async fn pending_count(&self) -> usize {
        self.pending_updates.read().await.len()
    }

    /// Clean up old pending updates (older than 1 hour).
    pub async fn cleanup_pending(&self) {
        let cutoff = Utc::now() - chrono::Duration::hours(1);
        let mut pending = self.pending_updates.write().await;
        pending.retain(|_, entry| entry.update.timestamp > cutoff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::config::AuditConfig;

    #[tokio::test]
    async fn test_push_and_subscribe() {
        let audit = Arc::new(AuditClient::new(AuditConfig::default()));
        let service = ConfigServiceImpl::new(audit);

        let mut subscriber = service.subscribe();

        let update = ConfigUpdateMessage {
            update_id: "update-1".to_string(),
            initiated_by: "test".to_string(),
            reason: "test update".to_string(),
            update_payload: serde_json::json!({"key": "value"}),
            timestamp: Utc::now(),
        };

        service.push_update(update).await.unwrap();

        let received = subscriber.recv().await.unwrap();
        assert_eq!(received.update_id, "update-1");
    }

    #[tokio::test]
    async fn test_process_ack() {
        let audit = Arc::new(AuditClient::new(AuditConfig::default()));
        let service = ConfigServiceImpl::new(audit);

        let update = ConfigUpdateMessage {
            update_id: "update-2".to_string(),
            initiated_by: "test".to_string(),
            reason: "test".to_string(),
            update_payload: serde_json::json!({}),
            timestamp: Utc::now(),
        };

        service.push_update(update).await.unwrap();

        let ack = ConfigAckMessage {
            update_id: "update-2".to_string(),
            applied: true,
            component_id: "agent-1".to_string(),
            error: None,
            config_version: 1,
        };

        service.process_ack(ack).await;
        assert_eq!(service.pending_count().await, 1);
    }
}
