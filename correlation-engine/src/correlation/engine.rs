//! Core correlation engine orchestrating the event processing pipeline.
//!
//! The engine is the central coordinator that:
//! 1. Receives events from the gRPC ingestion layer
//! 2. Adds events to the temporal window for the relevant GPU
//! 3. Runs the pattern matcher on the current window
//! 4. Updates the Bayesian belief
//! 5. Checks thresholds and triggers state transitions
//! 6. Emits quarantine directives and audit events on state changes
//!
//! The engine uses a per-GPU actor model via Tokio channels to ensure
//! events for the same GPU are processed sequentially without global locks.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::Utc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

use crate::correlation::bayesian_attribution::{BayesianAttributor, ReliabilityTier};
use crate::correlation::causal_model::CausalModel;
use crate::correlation::pattern_matcher::{DetectedPattern, PatternMatcher};
use crate::correlation::temporal_window::{
    CorrelationEvent, EventRingBuffer, EventType, TemporalWindow,
};
use crate::health::quarantine::{QuarantineAction, QuarantineManager, QuarantineState};
use crate::util::config::AppConfig;
use crate::util::metrics;

/// Commands sent to the correlation engine actor.
#[derive(Debug)]
pub enum EngineCommand {
    /// Process an incoming event.
    ProcessEvent(CorrelationEvent),

    /// Request a snapshot of GPU health (for queries).
    QueryGpuHealth {
        gpu_uuid: String,
        response: tokio::sync::oneshot::Sender<Option<GpuHealthSnapshot>>,
    },

    /// Request a fleet health summary.
    QueryFleetHealth {
        response: tokio::sync::oneshot::Sender<FleetHealthSnapshot>,
    },

    /// Request event history for a GPU.
    QueryGpuHistory {
        gpu_uuid: String,
        limit: usize,
        response: tokio::sync::oneshot::Sender<Vec<CorrelationEvent>>,
    },

    /// Shut down the engine.
    Shutdown,
}

/// A snapshot of a GPU's health state for query responses.
#[derive(Debug, Clone)]
pub struct GpuHealthSnapshot {
    pub gpu_uuid: String,
    pub state: QuarantineState,
    pub reliability_score: f64,
    pub alpha: f64,
    pub beta: f64,
    pub probe_pass_count: u64,
    pub probe_fail_count: u64,
    pub anomaly_count: u64,
    pub recent_patterns: Vec<DetectedPattern>,
}

/// A snapshot of fleet-wide health for query responses.
#[derive(Debug, Clone)]
pub struct FleetHealthSnapshot {
    pub total_gpus: u32,
    pub healthy: u32,
    pub suspect: u32,
    pub quarantined: u32,
    pub deep_test: u32,
    pub condemned: u32,
    pub average_reliability: f64,
}

/// The output of processing a single event through the correlation pipeline.
#[derive(Debug)]
pub struct CorrelationResult {
    /// The event that was processed.
    pub event: CorrelationEvent,

    /// Patterns detected as a result of this event.
    pub patterns: Vec<DetectedPattern>,

    /// The new reliability tier for the GPU.
    pub tier: ReliabilityTier,

    /// Whether a state transition occurred.
    pub state_changed: bool,

    /// The quarantine action to emit, if any.
    pub quarantine_action: Option<QuarantineAction>,
}

/// Handle for sending commands to the correlation engine.
#[derive(Clone)]
pub struct EngineHandle {
    sender: mpsc::Sender<EngineCommand>,
}

impl EngineHandle {
    /// Submit an event for correlation processing.
    pub async fn process_event(&self, event: CorrelationEvent) -> Result<()> {
        self.sender
            .send(EngineCommand::ProcessEvent(event))
            .await
            .context("engine channel closed")?;
        Ok(())
    }

    /// Query the health state of a specific GPU.
    pub async fn query_gpu_health(&self, gpu_uuid: String) -> Result<Option<GpuHealthSnapshot>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.sender
            .send(EngineCommand::QueryGpuHealth {
                gpu_uuid,
                response: tx,
            })
            .await
            .context("engine channel closed")?;
        rx.await.context("engine dropped response channel")
    }

    /// Query fleet-wide health.
    pub async fn query_fleet_health(&self) -> Result<FleetHealthSnapshot> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.sender
            .send(EngineCommand::QueryFleetHealth { response: tx })
            .await
            .context("engine channel closed")?;
        rx.await.context("engine dropped response channel")
    }

    /// Query event history for a GPU.
    pub async fn query_gpu_history(
        &self,
        gpu_uuid: String,
        limit: usize,
    ) -> Result<Vec<CorrelationEvent>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.sender
            .send(EngineCommand::QueryGpuHistory {
                gpu_uuid,
                limit,
                response: tx,
            })
            .await
            .context("engine channel closed")?;
        rx.await.context("engine dropped response channel")
    }

    /// Request engine shutdown.
    pub async fn shutdown(&self) -> Result<()> {
        self.sender
            .send(EngineCommand::Shutdown)
            .await
            .context("engine channel closed")?;
        Ok(())
    }
}

/// Channel for receiving quarantine actions emitted by the engine.
pub type QuarantineActionReceiver = mpsc::Receiver<(QuarantineAction, String, Vec<String>)>;

/// The correlation engine actor.
pub struct CorrelationEngine {
    /// Application configuration.
    config: Arc<AppConfig>,

    /// Bayesian attribution model.
    attributor: Arc<BayesianAttributor>,

    /// Pattern matcher.
    pattern_matcher: PatternMatcher,

    /// Causal model for environmental factors.
    causal_model: Arc<RwLock<CausalModel>>,

    /// Temporal sliding window.
    temporal_window: TemporalWindow,

    /// In-memory event ring buffer.
    ring_buffer: EventRingBuffer,

    /// Quarantine manager.
    quarantine_manager: Arc<RwLock<QuarantineManager>>,

    /// Channel for emitting quarantine actions.
    quarantine_tx: mpsc::Sender<(QuarantineAction, String, Vec<String>)>,

    /// Recent patterns per GPU (for query responses).
    recent_patterns: HashMap<String, Vec<DetectedPattern>>,
}

impl CorrelationEngine {
    /// Create a new correlation engine and return a handle for sending commands
    /// and a receiver for quarantine actions.
    pub fn new(
        config: Arc<AppConfig>,
        quarantine_manager: Arc<RwLock<QuarantineManager>>,
    ) -> (EngineHandle, QuarantineActionReceiver, mpsc::Receiver<EngineCommand>) {
        let (cmd_tx, cmd_rx) = mpsc::channel(10_000);
        let (q_tx, q_rx) = mpsc::channel(1_000);

        let handle = EngineHandle { sender: cmd_tx };

        // The caller will spawn the engine loop using `run`.
        // We return the command receiver so the caller can pass it to `run`.
        (handle, q_rx, cmd_rx)
    }

    /// Create the engine instance (called internally).
    fn create(
        config: Arc<AppConfig>,
        quarantine_manager: Arc<RwLock<QuarantineManager>>,
        quarantine_tx: mpsc::Sender<(QuarantineAction, String, Vec<String>)>,
    ) -> Self {
        let bayesian_config = config.bayesian.clone();
        let window_duration =
            Duration::from_secs(config.temporal_window.window_duration_secs);
        let ring_buffer_size = config.event_store.ring_buffer_size;

        Self {
            config: config.clone(),
            attributor: Arc::new(BayesianAttributor::new(bayesian_config)),
            pattern_matcher: PatternMatcher::with_defaults(),
            causal_model: Arc::new(RwLock::new(CausalModel::new())),
            temporal_window: TemporalWindow::new(window_duration),
            ring_buffer: EventRingBuffer::new(ring_buffer_size),
            quarantine_manager,
            quarantine_tx,
            recent_patterns: HashMap::new(),
        }
    }

    /// Run the correlation engine event loop. This consumes the engine.
    pub async fn run(
        config: Arc<AppConfig>,
        quarantine_manager: Arc<RwLock<QuarantineManager>>,
        quarantine_tx: mpsc::Sender<(QuarantineAction, String, Vec<String>)>,
        mut cmd_rx: mpsc::Receiver<EngineCommand>,
    ) {
        let mut engine = Self::create(config.clone(), quarantine_manager, quarantine_tx);

        // Spawn a periodic cleanup task for the temporal window.
        let cleanup_interval =
            Duration::from_secs(config.temporal_window.cleanup_interval_secs);

        let mut cleanup_timer = tokio::time::interval(cleanup_interval);
        cleanup_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        info!("Correlation engine started");

        loop {
            tokio::select! {
                Some(cmd) = cmd_rx.recv() => {
                    match cmd {
                        EngineCommand::ProcessEvent(event) => {
                            engine.handle_event(event).await;
                        }
                        EngineCommand::QueryGpuHealth { gpu_uuid, response } => {
                            let snapshot = engine.build_gpu_health_snapshot(&gpu_uuid);
                            let _ = response.send(snapshot);
                        }
                        EngineCommand::QueryFleetHealth { response } => {
                            let snapshot = engine.build_fleet_health_snapshot().await;
                            let _ = response.send(snapshot);
                        }
                        EngineCommand::QueryGpuHistory { gpu_uuid, limit, response } => {
                            let events = engine.ring_buffer.recent_events(&gpu_uuid, limit)
                                .into_iter()
                                .cloned()
                                .collect();
                            let _ = response.send(events);
                        }
                        EngineCommand::Shutdown => {
                            info!("Correlation engine shutting down");
                            break;
                        }
                    }
                }
                _ = cleanup_timer.tick() => {
                    let expired = engine.temporal_window.expire_old_events();
                    if expired > 0 {
                        debug!(expired, "Expired old events from temporal window");
                    }
                }
            }
        }
    }

    /// Get a reference to the Bayesian attributor (for sharing with other components).
    pub fn attributor(&self) -> &Arc<BayesianAttributor> {
        &self.attributor
    }

    /// Process a single event through the full correlation pipeline.
    async fn handle_event(&mut self, event: CorrelationEvent) {
        let start = std::time::Instant::now();
        let event_type_str = event.event_type.to_string();
        let gpu_uuid = event.gpu_uuid.clone();

        // Step 1: Add to temporal window.
        self.temporal_window.insert(event.clone());

        // Step 2: Add to ring buffer for historical queries.
        self.ring_buffer.push(event.clone());

        // Step 3: Update Bayesian belief.
        let tier = match event.event_type {
            EventType::ProbePass => {
                self.attributor
                    .record_probe_success(&gpu_uuid, event.sm_id)
            }
            EventType::ProbeFail => {
                self.attributor
                    .record_probe_failure(&gpu_uuid, event.sm_id)
            }
            EventType::ProbeError => {
                // Probe errors are less informative; treat as a mild negative signal.
                self.attributor.record_anomaly(&gpu_uuid)
            }
            EventType::InferenceAnomaly
            | EventType::TrainingAnomaly
            | EventType::InvariantViolation => self.attributor.record_anomaly(&gpu_uuid),
            EventType::ThermalSpike | EventType::PowerAnomaly | EventType::EccError => {
                // Environmental events update causal model but don't directly
                // affect the Bayesian belief. Return current tier.
                self.attributor.gpu_tier(&gpu_uuid)
            }
        };

        // Step 4: Update metrics.
        if let Some(score) = self.attributor.gpu_reliability_score(&gpu_uuid) {
            metrics::set_gpu_health_score(&gpu_uuid, score);
        }

        // Step 5: Run pattern matcher on current window.
        let gpu_events = self.temporal_window.events_for_gpu(&gpu_uuid);
        let patterns = self.pattern_matcher.match_patterns(&gpu_uuid, &gpu_events);

        if !patterns.is_empty() {
            for p in &patterns {
                metrics::record_pattern_detected(&p.pattern_type.to_string());
                debug!(
                    gpu = %gpu_uuid,
                    pattern = %p.pattern_type,
                    confidence = p.confidence,
                    "Pattern detected"
                );
            }
            self.recent_patterns
                .insert(gpu_uuid.clone(), patterns.clone());
        }

        // Step 6: Check thresholds and trigger state transitions.
        self.check_and_transition(&gpu_uuid, tier, &patterns).await;

        // Step 7: Record latency metric.
        let elapsed = start.elapsed().as_secs_f64();
        metrics::observe_correlation_latency(&event_type_str, elapsed);
    }

    /// Check if the current tier warrants a state transition and execute it.
    async fn check_and_transition(
        &self,
        gpu_uuid: &str,
        tier: ReliabilityTier,
        patterns: &[DetectedPattern],
    ) {
        let mut qm = self.quarantine_manager.write().await;
        let current_state = qm.get_state(gpu_uuid);

        let new_state = match tier {
            ReliabilityTier::Healthy => QuarantineState::Healthy,
            ReliabilityTier::IncreasedProbing => {
                // Don't change state, just increase probing frequency externally.
                return;
            }
            ReliabilityTier::Suspect => QuarantineState::Suspect,
            ReliabilityTier::Quarantine => QuarantineState::Quarantined,
            ReliabilityTier::Condemned => QuarantineState::Condemned,
        };

        // Only transition if the new state is "worse" or explicitly returned to healthy.
        let should_transition = match (current_state, new_state) {
            (QuarantineState::Healthy, QuarantineState::Suspect)
            | (QuarantineState::Healthy, QuarantineState::Quarantined)
            | (QuarantineState::Healthy, QuarantineState::Condemned)
            | (QuarantineState::Suspect, QuarantineState::Quarantined)
            | (QuarantineState::Suspect, QuarantineState::Condemned)
            | (QuarantineState::Quarantined, QuarantineState::Condemned)
            | (QuarantineState::DeepTest, QuarantineState::Condemned) => true,
            _ => false,
        };

        if should_transition {
            let evidence: Vec<String> = patterns.iter().map(|p| p.pattern_id.clone()).collect();
            let reason = format!(
                "Bayesian tier {} triggered state transition from {:?} to {:?}",
                tier, current_state, new_state
            );

            let action = match new_state {
                QuarantineState::Suspect => QuarantineAction::MarkSuspect,
                QuarantineState::Quarantined => QuarantineAction::Quarantine,
                QuarantineState::DeepTest => QuarantineAction::ScheduleDeepTest,
                QuarantineState::Condemned => QuarantineAction::Condemn,
                QuarantineState::Healthy => QuarantineAction::Reinstate,
            };

            if let Err(e) = qm.transition(gpu_uuid, action, &reason) {
                warn!(
                    gpu = %gpu_uuid,
                    error = %e,
                    "Failed to transition GPU state"
                );
                return;
            }

            info!(
                gpu = %gpu_uuid,
                from = ?current_state,
                to = ?new_state,
                reason = %reason,
                "GPU state transition"
            );

            metrics::record_quarantine_action(&format!("{:?}", action));

            // Emit quarantine action for downstream processing.
            if let Err(e) = self
                .quarantine_tx
                .send((action, gpu_uuid.to_string(), evidence))
                .await
            {
                error!(
                    error = %e,
                    "Failed to emit quarantine action"
                );
            }
        }
    }

    /// Build a GPU health snapshot for query responses.
    fn build_gpu_health_snapshot(&self, gpu_uuid: &str) -> Option<GpuHealthSnapshot> {
        let belief = self.attributor.gpu_belief(gpu_uuid)?;

        let state = {
            // We need a synchronous read here. Use try_read since we're
            // in an async context but called from within the actor loop.
            match self.quarantine_manager.try_read() {
                Ok(qm) => qm.get_state(gpu_uuid),
                Err(_) => QuarantineState::Healthy,
            }
        };

        let recent_patterns = self
            .recent_patterns
            .get(gpu_uuid)
            .cloned()
            .unwrap_or_default();

        Some(GpuHealthSnapshot {
            gpu_uuid: gpu_uuid.to_string(),
            state,
            reliability_score: belief.reliability_score(),
            alpha: belief.alpha,
            beta: belief.beta,
            probe_pass_count: belief.total_successes,
            probe_fail_count: belief.total_failures,
            anomaly_count: belief.total_anomalies,
            recent_patterns,
        })
    }

    /// Build a fleet health snapshot.
    async fn build_fleet_health_snapshot(&self) -> FleetHealthSnapshot {
        let qm = self.quarantine_manager.read().await;
        let all_gpus = self.attributor.tracked_gpus();
        let total = all_gpus.len() as u32;

        let mut healthy = 0u32;
        let mut suspect = 0u32;
        let mut quarantined = 0u32;
        let mut deep_test = 0u32;
        let mut condemned = 0u32;
        let mut reliability_sum = 0.0f64;

        for gpu_uuid in &all_gpus {
            let state = qm.get_state(gpu_uuid);
            match state {
                QuarantineState::Healthy => healthy += 1,
                QuarantineState::Suspect => suspect += 1,
                QuarantineState::Quarantined => quarantined += 1,
                QuarantineState::DeepTest => deep_test += 1,
                QuarantineState::Condemned => condemned += 1,
            }
            if let Some(score) = self.attributor.gpu_reliability_score(gpu_uuid) {
                reliability_sum += score;
            }
        }

        let average_reliability = if total > 0 {
            reliability_sum / total as f64
        } else {
            1.0
        };

        // Update metrics.
        metrics::set_gpu_state_count("healthy", healthy as i64);
        metrics::set_gpu_state_count("suspect", suspect as i64);
        metrics::set_gpu_state_count("quarantined", quarantined as i64);
        metrics::set_gpu_state_count("deep_test", deep_test as i64);
        metrics::set_gpu_state_count("condemned", condemned as i64);

        FleetHealthSnapshot {
            total_gpus: total,
            healthy,
            suspect,
            quarantined,
            deep_test,
            condemned,
            average_reliability,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::quarantine::QuarantineManager;

    fn test_config() -> Arc<AppConfig> {
        Arc::new(AppConfig::default())
    }

    #[tokio::test]
    async fn test_engine_processes_probe_pass() {
        let config = test_config();
        let qm = Arc::new(RwLock::new(QuarantineManager::new(
            config.quarantine.clone(),
        )));
        let (handle, _q_rx, cmd_rx) = CorrelationEngine::new(config.clone(), qm.clone());

        let engine_task = tokio::spawn(async move {
            let (q_tx, _) = mpsc::channel(100);
            CorrelationEngine::run(config, qm, q_tx, cmd_rx).await;
        });

        let event = CorrelationEvent {
            event_id: "test-1".to_string(),
            gpu_uuid: "gpu-1".to_string(),
            sm_id: None,
            event_type: EventType::ProbePass,
            timestamp: Utc::now(),
            severity: 1,
            score: 0.0,
            metadata: HashMap::new(),
        };

        handle.process_event(event).await.unwrap();

        // Give the engine a moment to process.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let snapshot = handle.query_gpu_health("gpu-1".to_string()).await.unwrap();
        assert!(snapshot.is_some());
        let snap = snapshot.unwrap();
        assert_eq!(snap.probe_pass_count, 1);
        assert!(snap.reliability_score > 0.999);

        handle.shutdown().await.unwrap();
        let _ = engine_task.await;
    }
}
