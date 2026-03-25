//! Event persistence layer.
//!
//! Provides an abstracted interface for persisting correlation events. In
//! development mode, events are stored in an in-memory ring buffer. In
//! production, events would be written to ScyllaDB or a similar persistent
//! store.
//!
//! The event store supports:
//! - Batched writes (configurable flush interval and batch size)
//! - Per-GPU ring buffer (last N events in memory for fast queries)
//! - Event retrieval by GPU UUID and time range

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info};

use crate::correlation::temporal_window::CorrelationEvent;
use crate::util::config::EventStoreConfig;

/// Trait for event persistence backends.
#[async_trait::async_trait]
pub trait EventPersistence: Send + Sync {
    /// Write a batch of events to the persistent store.
    async fn write_batch(&self, events: &[CorrelationEvent]) -> Result<()>;

    /// Query events for a GPU within a time range.
    async fn query_events(
        &self,
        gpu_uuid: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<CorrelationEvent>>;
}

/// In-memory event persistence backend for development and testing.
pub struct InMemoryPersistence {
    /// Per-GPU event storage with bounded capacity.
    events: Mutex<HashMap<String, VecDeque<CorrelationEvent>>>,
    /// Maximum events per GPU.
    max_per_gpu: usize,
}

impl InMemoryPersistence {
    /// Create a new in-memory persistence backend.
    pub fn new(max_per_gpu: usize) -> Self {
        Self {
            events: Mutex::new(HashMap::new()),
            max_per_gpu,
        }
    }
}

#[async_trait::async_trait]
impl EventPersistence for InMemoryPersistence {
    async fn write_batch(&self, events: &[CorrelationEvent]) -> Result<()> {
        let mut store = self.events.lock().await;
        for event in events {
            let buf = store.entry(event.gpu_uuid.clone()).or_default();
            if buf.len() >= self.max_per_gpu {
                buf.pop_front();
            }
            buf.push_back(event.clone());
        }
        Ok(())
    }

    async fn query_events(
        &self,
        gpu_uuid: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: usize,
    ) -> Result<Vec<CorrelationEvent>> {
        let store = self.events.lock().await;
        let events = store
            .get(gpu_uuid)
            .map(|buf| {
                buf.iter()
                    .filter(|e| e.timestamp >= start && e.timestamp <= end)
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        Ok(events)
    }
}

/// The event store manages batched writes to a persistent backend.
///
/// Events are buffered in memory and flushed periodically or when the
/// batch reaches a configurable maximum size.
pub struct EventStore {
    /// Sender for submitting events to the batch writer.
    sender: mpsc::Sender<CorrelationEvent>,
}

impl EventStore {
    /// Create a new event store with the given configuration and persistence backend.
    ///
    /// Returns the event store handle and a task handle for the background writer.
    pub fn new(
        config: EventStoreConfig,
        persistence: Arc<dyn EventPersistence>,
    ) -> (Self, tokio::task::JoinHandle<()>) {
        let (tx, rx) = mpsc::channel(10_000);

        let handle = tokio::spawn(Self::batch_writer_loop(config, persistence, rx));

        (Self { sender: tx }, handle)
    }

    /// Submit an event for batched persistence.
    pub async fn write(&self, event: CorrelationEvent) -> Result<()> {
        self.sender
            .send(event)
            .await
            .map_err(|_| anyhow::anyhow!("event store channel closed"))?;
        Ok(())
    }

    /// Background loop that buffers events and flushes them periodically.
    async fn batch_writer_loop(
        config: EventStoreConfig,
        persistence: Arc<dyn EventPersistence>,
        mut rx: mpsc::Receiver<CorrelationEvent>,
    ) {
        let flush_interval = Duration::from_millis(config.flush_interval_ms);
        let max_batch = config.max_batch_size;
        let mut batch: Vec<CorrelationEvent> = Vec::with_capacity(max_batch);
        let mut interval = tokio::time::interval(flush_interval);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        info!(
            flush_ms = config.flush_interval_ms,
            max_batch = max_batch,
            "Event store batch writer started"
        );

        loop {
            tokio::select! {
                Some(event) = rx.recv() => {
                    batch.push(event);
                    if batch.len() >= max_batch {
                        Self::flush_batch(&persistence, &mut batch).await;
                    }
                }
                _ = interval.tick() => {
                    if !batch.is_empty() {
                        Self::flush_batch(&persistence, &mut batch).await;
                    }
                }
                else => {
                    // Channel closed; flush remaining events and exit.
                    if !batch.is_empty() {
                        Self::flush_batch(&persistence, &mut batch).await;
                    }
                    info!("Event store batch writer shutting down");
                    break;
                }
            }
        }
    }

    /// Flush the accumulated batch to the persistence backend.
    async fn flush_batch(
        persistence: &Arc<dyn EventPersistence>,
        batch: &mut Vec<CorrelationEvent>,
    ) {
        let count = batch.len();
        match persistence.write_batch(batch).await {
            Ok(()) => {
                debug!(count, "Flushed event batch to persistence");
            }
            Err(e) => {
                error!(
                    count,
                    error = %e,
                    "Failed to flush event batch; events may be lost"
                );
            }
        }
        batch.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlation::temporal_window::EventType;

    fn make_event(gpu: &str) -> CorrelationEvent {
        CorrelationEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            gpu_uuid: gpu.to_string(),
            sm_id: None,
            event_type: EventType::ProbePass,
            timestamp: Utc::now(),
            severity: 1,
            score: 0.0,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_in_memory_persistence() {
        let persistence = InMemoryPersistence::new(100);
        let events = vec![make_event("gpu-1"), make_event("gpu-1"), make_event("gpu-2")];

        persistence.write_batch(&events).await.unwrap();

        let result = persistence
            .query_events(
                "gpu-1",
                Utc::now() - chrono::Duration::hours(1),
                Utc::now() + chrono::Duration::hours(1),
                10,
            )
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_in_memory_ring_buffer_eviction() {
        let persistence = InMemoryPersistence::new(2);
        let events = vec![
            make_event("gpu-1"),
            make_event("gpu-1"),
            make_event("gpu-1"),
        ];

        persistence.write_batch(&events).await.unwrap();

        let result = persistence
            .query_events(
                "gpu-1",
                Utc::now() - chrono::Duration::hours(1),
                Utc::now() + chrono::Duration::hours(1),
                10,
            )
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_event_store_batching() {
        let persistence = Arc::new(InMemoryPersistence::new(1000));
        let config = EventStoreConfig {
            ring_buffer_size: 1000,
            flush_interval_ms: 50,
            max_batch_size: 100,
        };

        let (store, writer_handle) = EventStore::new(config, persistence.clone());

        for _ in 0..5 {
            store.write(make_event("gpu-1")).await.unwrap();
        }

        // Wait for flush.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let result = persistence
            .query_events(
                "gpu-1",
                Utc::now() - chrono::Duration::hours(1),
                Utc::now() + chrono::Duration::hours(1),
                10,
            )
            .await
            .unwrap();
        assert_eq!(result.len(), 5);

        drop(store);
        let _ = writer_handle.await;
    }
}
