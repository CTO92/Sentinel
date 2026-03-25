//! Alert routing, deduplication, and rate limiting.
//!
//! The alert manager receives alert requests from the correlation engine and
//! dispatches them to configured channels (PagerDuty, Slack, webhooks) while
//! enforcing rate limits and deduplication rules.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, error, info};

use crate::alerting::pagerduty::PagerDutyClient;
use crate::alerting::slack::SlackClient;
use crate::alerting::webhook::WebhookClient;
use crate::util::config::AlertingConfig;
use crate::util::metrics;

/// Severity levels for alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    High,
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "info"),
            AlertSeverity::Warning => write!(f, "warning"),
            AlertSeverity::High => write!(f, "high"),
            AlertSeverity::Critical => write!(f, "critical"),
        }
    }
}

/// An alert to be dispatched to notification channels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique alert identifier.
    pub alert_id: String,

    /// Alert severity.
    pub severity: AlertSeverity,

    /// Short summary of the alert.
    pub summary: String,

    /// Detailed description.
    pub description: String,

    /// GPU UUID (if GPU-specific).
    pub gpu_uuid: Option<String>,

    /// Hostname (if host-specific).
    pub hostname: Option<String>,

    /// Alert source component.
    pub source: String,

    /// Deduplication key for rate limiting.
    pub dedup_key: String,

    /// When the alert was generated.
    pub timestamp: DateTime<Utc>,

    /// Additional context as key-value pairs.
    pub metadata: HashMap<String, String>,
}

/// Handle for submitting alerts.
#[derive(Clone)]
pub struct AlertHandle {
    sender: mpsc::Sender<Alert>,
}

impl AlertHandle {
    /// Submit an alert for dispatch.
    pub async fn send_alert(&self, alert: Alert) -> Result<()> {
        self.sender
            .send(alert)
            .await
            .map_err(|_| anyhow::anyhow!("alert channel closed"))?;
        Ok(())
    }
}

/// The alert manager dispatches alerts to configured channels.
pub struct AlertManager {
    /// Configuration.
    config: AlertingConfig,

    /// PagerDuty client.
    pagerduty: Option<PagerDutyClient>,

    /// Slack client.
    slack: Option<SlackClient>,

    /// Webhook clients.
    webhooks: Vec<WebhookClient>,

    /// Rate limiting state: dedup_key -> last sent time.
    rate_limiter: HashMap<String, Instant>,

    /// Global rate limiter: number of alerts sent in the current minute.
    global_count: u32,

    /// When the global count was last reset.
    global_reset_at: Instant,
}

impl AlertManager {
    /// Create a new alert manager and return a handle for submitting alerts.
    pub fn new(_config: AlertingConfig) -> (AlertHandle, mpsc::Receiver<Alert>) {
        let (tx, rx) = mpsc::channel(1_000);
        (AlertHandle { sender: tx }, rx)
    }

    /// Create the alert manager instance for the dispatch loop.
    pub fn create(config: AlertingConfig) -> Self {
        let pagerduty = if config.pagerduty.enabled {
            Some(PagerDutyClient::new(config.pagerduty.clone()))
        } else {
            None
        };

        let slack = if config.slack.enabled {
            Some(SlackClient::new(config.slack.clone()))
        } else {
            None
        };

        let webhooks: Vec<WebhookClient> = config
            .webhooks
            .iter()
            .filter(|w| w.enabled)
            .map(|w| WebhookClient::new(w.clone()))
            .collect();

        Self {
            config,
            pagerduty,
            slack,
            webhooks,
            rate_limiter: HashMap::new(),
            global_count: 0,
            global_reset_at: Instant::now(),
        }
    }

    /// Run the alert dispatch loop.
    pub async fn run(config: AlertingConfig, mut rx: mpsc::Receiver<Alert>) {
        let mut manager = Self::create(config);

        info!(
            pagerduty = manager.pagerduty.is_some(),
            slack = manager.slack.is_some(),
            webhooks = manager.webhooks.len(),
            "Alert manager started"
        );

        while let Some(alert) = rx.recv().await {
            if !manager.config.enabled {
                continue;
            }

            if manager.should_rate_limit(&alert) {
                debug!(
                    dedup_key = %alert.dedup_key,
                    "Alert rate-limited"
                );
                continue;
            }

            manager.dispatch(alert).await;
        }

        info!("Alert manager shutting down");
    }

    /// Check if an alert should be rate-limited.
    fn should_rate_limit(&mut self, alert: &Alert) -> bool {
        // Reset global counter every minute.
        if self.global_reset_at.elapsed() >= Duration::from_secs(60) {
            self.global_count = 0;
            self.global_reset_at = Instant::now();
        }

        // Check global rate limit.
        if self.global_count >= self.config.max_alerts_per_minute {
            return true;
        }

        // Check per-key rate limit.
        let rate_limit = Duration::from_secs(self.config.rate_limit_secs);
        if let Some(last_sent) = self.rate_limiter.get(&alert.dedup_key) {
            if last_sent.elapsed() < rate_limit {
                return true;
            }
        }

        false
    }

    /// Dispatch an alert to all configured channels.
    async fn dispatch(&mut self, alert: Alert) {
        // Record rate limiting state.
        self.rate_limiter
            .insert(alert.dedup_key.clone(), Instant::now());
        self.global_count += 1;

        // Dispatch to PagerDuty (only for High and Critical).
        if alert.severity >= AlertSeverity::High {
            if let Some(ref pd) = self.pagerduty {
                match pd.send_alert(&alert).await {
                    Ok(()) => metrics::record_alert_sent("pagerduty"),
                    Err(e) => error!(error = %e, "Failed to send PagerDuty alert"),
                }
            }
        }

        // Dispatch to Slack (all severities).
        if let Some(ref slack) = self.slack {
            match slack.send_alert(&alert).await {
                Ok(()) => metrics::record_alert_sent("slack"),
                Err(e) => error!(error = %e, "Failed to send Slack alert"),
            }
        }

        // Dispatch to webhooks.
        for webhook in &self.webhooks {
            match webhook.send_alert(&alert).await {
                Ok(()) => metrics::record_alert_sent("webhook"),
                Err(e) => error!(
                    webhook = %webhook.name(),
                    error = %e,
                    "Failed to send webhook alert"
                ),
            }
        }

        info!(
            alert_id = %alert.alert_id,
            severity = %alert.severity,
            summary = %alert.summary,
            "Alert dispatched"
        );
    }

    /// Clean up stale rate limiter entries (older than 2x the rate limit window).
    pub fn cleanup_rate_limiter(&mut self) {
        let max_age = Duration::from_secs(self.config.rate_limit_secs * 2);
        self.rate_limiter
            .retain(|_, last_sent| last_sent.elapsed() < max_age);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_alert(dedup_key: &str) -> Alert {
        Alert {
            alert_id: uuid::Uuid::new_v4().to_string(),
            severity: AlertSeverity::High,
            summary: "Test alert".to_string(),
            description: "This is a test alert".to_string(),
            gpu_uuid: Some("gpu-1".to_string()),
            hostname: Some("host-1".to_string()),
            source: "test".to_string(),
            dedup_key: dedup_key.to_string(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_rate_limiting() {
        let config = AlertingConfig {
            rate_limit_secs: 300,
            max_alerts_per_minute: 10,
            ..AlertingConfig::default()
        };
        let mut manager = AlertManager::create(config);

        let alert = test_alert("test-key");

        // First alert should not be rate-limited.
        assert!(!manager.should_rate_limit(&alert));

        // Record it as sent.
        manager
            .rate_limiter
            .insert(alert.dedup_key.clone(), Instant::now());

        // Same dedup key should be rate-limited.
        assert!(manager.should_rate_limit(&alert));

        // Different dedup key should not be rate-limited.
        let alert2 = test_alert("different-key");
        assert!(!manager.should_rate_limit(&alert2));
    }

    #[test]
    fn test_global_rate_limit() {
        let config = AlertingConfig {
            rate_limit_secs: 0, // No per-key rate limit.
            max_alerts_per_minute: 2,
            ..AlertingConfig::default()
        };
        let mut manager = AlertManager::create(config);

        manager.global_count = 2;
        let alert = test_alert("test");
        assert!(manager.should_rate_limit(&alert));
    }
}
