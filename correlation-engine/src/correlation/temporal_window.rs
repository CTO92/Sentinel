//! Temporal windowing for event correlation.
//!
//! The temporal window groups events by GPU within a configurable sliding time
//! window (default 5 minutes). Events are efficiently inserted and expired,
//! enabling the pattern matcher to operate on recent, co-occurring events.
//!
//! Implementation uses a sorted `BTreeMap` keyed by timestamp for O(log n)
//! insertion and efficient range-based expiration.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A unified event representation used within the correlation window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationEvent {
    /// Unique event identifier.
    pub event_id: String,

    /// GPU UUID this event is associated with.
    pub gpu_uuid: String,

    /// Optional SM identifier (for probe events localized to an SM).
    pub sm_id: Option<u32>,

    /// The type of event.
    pub event_type: EventType,

    /// Timestamp when the event occurred.
    pub timestamp: DateTime<Utc>,

    /// Severity level (1=INFO, 2=WARNING, 3=HIGH, 4=CRITICAL).
    pub severity: u32,

    /// Numeric score or magnitude (interpretation depends on event type).
    pub score: f64,

    /// Additional context as key-value pairs.
    pub metadata: HashMap<String, String>,
}

/// Classification of events flowing through the correlation pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// A probe execution that passed.
    ProbePass,
    /// A probe execution that failed (SDC detected).
    ProbeFail,
    /// A probe execution that timed out or errored.
    ProbeError,
    /// An inference anomaly event.
    InferenceAnomaly,
    /// A training anomaly event.
    TrainingAnomaly,
    /// An invariant violation event.
    InvariantViolation,
    /// A thermal spike event (derived from telemetry).
    ThermalSpike,
    /// A power anomaly event (derived from telemetry).
    PowerAnomaly,
    /// An ECC error event (derived from telemetry).
    EccError,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::ProbePass => write!(f, "probe_pass"),
            EventType::ProbeFail => write!(f, "probe_fail"),
            EventType::ProbeError => write!(f, "probe_error"),
            EventType::InferenceAnomaly => write!(f, "inference_anomaly"),
            EventType::TrainingAnomaly => write!(f, "training_anomaly"),
            EventType::InvariantViolation => write!(f, "invariant_violation"),
            EventType::ThermalSpike => write!(f, "thermal_spike"),
            EventType::PowerAnomaly => write!(f, "power_anomaly"),
            EventType::EccError => write!(f, "ecc_error"),
        }
    }
}

/// An ordered key for the BTreeMap, combining timestamp with a unique tiebreaker.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TimeKey {
    timestamp_ns: i64,
    sequence: u64,
}

/// A sliding temporal window that groups events by GPU.
///
/// Events are stored in a time-ordered structure per GPU. Old events are
/// automatically expired when the window is cleaned.
#[derive(Debug)]
pub struct TemporalWindow {
    /// The duration of the sliding window.
    window_duration: Duration,

    /// Per-GPU event storage. The inner BTreeMap is keyed by (timestamp_ns, seq)
    /// to maintain strict ordering even for events with identical timestamps.
    gpu_events: HashMap<String, BTreeMap<TimeKey, CorrelationEvent>>,

    /// Monotonically increasing sequence counter for tiebreaking.
    sequence: u64,

    /// Total number of events currently in the window.
    total_events: usize,
}

impl TemporalWindow {
    /// Create a new temporal window with the given duration.
    pub fn new(window_duration: Duration) -> Self {
        Self {
            window_duration,
            gpu_events: HashMap::new(),
            sequence: 0,
            total_events: 0,
        }
    }

    /// Insert an event into the temporal window.
    ///
    /// The event is indexed by its GPU UUID and timestamp.
    pub fn insert(&mut self, event: CorrelationEvent) {
        let key = TimeKey {
            timestamp_ns: event.timestamp.timestamp_nanos_opt().unwrap_or(0),
            sequence: self.sequence,
        };
        self.sequence += 1;

        self.gpu_events
            .entry(event.gpu_uuid.clone())
            .or_default()
            .insert(key, event);
        self.total_events += 1;
    }

    /// Get all events for a specific GPU within the current window.
    ///
    /// Returns events ordered by timestamp (oldest first).
    pub fn events_for_gpu(&self, gpu_uuid: &str) -> Vec<&CorrelationEvent> {
        let cutoff = Utc::now() - chrono::Duration::from_std(self.window_duration).unwrap();
        let cutoff_ns = cutoff.timestamp_nanos_opt().unwrap_or(0);

        match self.gpu_events.get(gpu_uuid) {
            Some(events) => events
                .range(TimeKey {
                    timestamp_ns: cutoff_ns,
                    sequence: 0,
                }..)
                .map(|(_, event)| event)
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get all events for a specific GPU of a specific type within the window.
    pub fn events_for_gpu_of_type(
        &self,
        gpu_uuid: &str,
        event_type: EventType,
    ) -> Vec<&CorrelationEvent> {
        self.events_for_gpu(gpu_uuid)
            .into_iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Get all events within the current window across all GPUs,
    /// useful for detecting cross-GPU correlation patterns.
    pub fn all_recent_events(&self) -> Vec<&CorrelationEvent> {
        let cutoff = Utc::now() - chrono::Duration::from_std(self.window_duration).unwrap();
        let cutoff_ns = cutoff.timestamp_nanos_opt().unwrap_or(0);

        let mut result = Vec::new();
        for events in self.gpu_events.values() {
            for (_, event) in events.range(TimeKey {
                timestamp_ns: cutoff_ns,
                sequence: 0,
            }..) {
                result.push(event);
            }
        }
        result.sort_by_key(|e| e.timestamp);
        result
    }

    /// Get all GPU UUIDs that have events in the current window.
    pub fn active_gpus(&self) -> Vec<String> {
        self.gpu_events.keys().cloned().collect()
    }

    /// Get events on the same host (by matching hostname in metadata).
    pub fn events_on_host(&self, hostname: &str) -> Vec<&CorrelationEvent> {
        let cutoff = Utc::now() - chrono::Duration::from_std(self.window_duration).unwrap();
        let cutoff_ns = cutoff.timestamp_nanos_opt().unwrap_or(0);

        let mut result = Vec::new();
        for events in self.gpu_events.values() {
            for (_, event) in events.range(TimeKey {
                timestamp_ns: cutoff_ns,
                sequence: 0,
            }..) {
                if event.metadata.get("hostname").map(|h| h.as_str()) == Some(hostname) {
                    result.push(event);
                }
            }
        }
        result
    }

    /// Expire events older than the window duration.
    ///
    /// Call this periodically (e.g., every 10 seconds) to reclaim memory.
    /// Returns the number of events expired.
    pub fn expire_old_events(&mut self) -> usize {
        let cutoff = Utc::now() - chrono::Duration::from_std(self.window_duration).unwrap();
        let cutoff_ns = cutoff.timestamp_nanos_opt().unwrap_or(0);
        let cutoff_key = TimeKey {
            timestamp_ns: cutoff_ns,
            sequence: u64::MAX,
        };

        let mut expired = 0;
        let mut empty_gpus = Vec::new();

        for (gpu_uuid, events) in self.gpu_events.iter_mut() {
            // Split off events before the cutoff.
            let remaining = events.split_off(&cutoff_key);
            expired += events.len();
            *events = remaining;

            if events.is_empty() {
                empty_gpus.push(gpu_uuid.clone());
            }
        }

        // Remove empty GPU entries to avoid unbounded HashMap growth.
        for gpu_uuid in empty_gpus {
            self.gpu_events.remove(&gpu_uuid);
        }

        self.total_events -= expired;
        expired
    }

    /// Get the total number of events currently in the window.
    pub fn len(&self) -> usize {
        self.total_events
    }

    /// Check if the window is empty.
    pub fn is_empty(&self) -> bool {
        self.total_events == 0
    }

    /// Get the number of GPUs with events in the window.
    pub fn gpu_count(&self) -> usize {
        self.gpu_events.len()
    }

    /// Get event counts per event type for a GPU within the window.
    pub fn event_type_counts(&self, gpu_uuid: &str) -> HashMap<EventType, usize> {
        let mut counts = HashMap::new();
        for event in self.events_for_gpu(gpu_uuid) {
            *counts.entry(event.event_type).or_insert(0) += 1;
        }
        counts
    }
}

/// A bounded ring buffer that stores the last N events per GPU,
/// used for the in-memory portion of the event store.
#[derive(Debug)]
pub struct EventRingBuffer {
    /// Maximum capacity.
    capacity: usize,

    /// Per-GPU ring buffers.
    buffers: HashMap<String, VecDeque<CorrelationEvent>>,
}

impl EventRingBuffer {
    /// Create a new ring buffer with the given per-GPU capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffers: HashMap::new(),
        }
    }

    /// Push an event into the ring buffer for its GPU.
    /// If the buffer is at capacity, the oldest event is evicted.
    pub fn push(&mut self, event: CorrelationEvent) {
        let buf = self.buffers.entry(event.gpu_uuid.clone()).or_default();
        if buf.len() >= self.capacity {
            buf.pop_front();
        }
        buf.push_back(event);
    }

    /// Get the last N events for a GPU (most recent last).
    pub fn recent_events(&self, gpu_uuid: &str, n: usize) -> Vec<&CorrelationEvent> {
        match self.buffers.get(gpu_uuid) {
            Some(buf) => {
                let start = buf.len().saturating_sub(n);
                buf.range(start..).collect()
            }
            None => Vec::new(),
        }
    }

    /// Get all events for a GPU.
    pub fn all_events(&self, gpu_uuid: &str) -> Vec<&CorrelationEvent> {
        match self.buffers.get(gpu_uuid) {
            Some(buf) => buf.iter().collect(),
            None => Vec::new(),
        }
    }

    /// Get the total number of events across all GPUs.
    pub fn total_events(&self) -> usize {
        self.buffers.values().map(|b| b.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(gpu: &str, event_type: EventType, age_secs: i64) -> CorrelationEvent {
        CorrelationEvent {
            event_id: uuid::Uuid::new_v4().to_string(),
            gpu_uuid: gpu.to_string(),
            sm_id: None,
            event_type,
            timestamp: Utc::now() - chrono::Duration::seconds(age_secs),
            severity: 2,
            score: 1.0,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_insert_and_retrieve() {
        let mut window = TemporalWindow::new(Duration::from_secs(300));
        window.insert(make_event("gpu-1", EventType::ProbeFail, 10));
        window.insert(make_event("gpu-1", EventType::InferenceAnomaly, 5));
        window.insert(make_event("gpu-2", EventType::ProbeFail, 3));

        assert_eq!(window.len(), 3);
        assert_eq!(window.events_for_gpu("gpu-1").len(), 2);
        assert_eq!(window.events_for_gpu("gpu-2").len(), 1);
        assert_eq!(window.events_for_gpu("gpu-3").len(), 0);
    }

    #[test]
    fn test_type_filtering() {
        let mut window = TemporalWindow::new(Duration::from_secs(300));
        window.insert(make_event("gpu-1", EventType::ProbeFail, 10));
        window.insert(make_event("gpu-1", EventType::ProbePass, 5));
        window.insert(make_event("gpu-1", EventType::InferenceAnomaly, 3));

        let fails = window.events_for_gpu_of_type("gpu-1", EventType::ProbeFail);
        assert_eq!(fails.len(), 1);
    }

    #[test]
    fn test_expire_old_events() {
        let mut window = TemporalWindow::new(Duration::from_secs(60));
        // Insert an event that is 120 seconds old (outside the 60s window).
        window.insert(make_event("gpu-1", EventType::ProbeFail, 120));
        // Insert a recent event.
        window.insert(make_event("gpu-1", EventType::ProbePass, 5));

        let expired = window.expire_old_events();
        assert_eq!(expired, 1);
        assert_eq!(window.len(), 1);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut rb = EventRingBuffer::new(3);
        for i in 0..5 {
            let mut event = make_event("gpu-1", EventType::ProbePass, 0);
            event.event_id = format!("event-{}", i);
            rb.push(event);
        }
        let events = rb.all_events("gpu-1");
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_id, "event-2");
        assert_eq!(events[2].event_id, "event-4");
    }

    #[test]
    fn test_event_type_counts() {
        let mut window = TemporalWindow::new(Duration::from_secs(300));
        window.insert(make_event("gpu-1", EventType::ProbeFail, 10));
        window.insert(make_event("gpu-1", EventType::ProbeFail, 8));
        window.insert(make_event("gpu-1", EventType::InferenceAnomaly, 5));

        let counts = window.event_type_counts("gpu-1");
        assert_eq!(counts.get(&EventType::ProbeFail), Some(&2));
        assert_eq!(counts.get(&EventType::InferenceAnomaly), Some(&1));
    }
}
