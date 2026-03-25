//! PagerDuty Events API v2 integration.
//!
//! Sends trigger events to PagerDuty when GPUs are quarantined or condemned.
//! Supports automatic resolve when GPUs are reinstated.

use anyhow::{Context, Result};
use serde_json::json;
use tracing::debug;

use crate::alerting::alert_manager::{Alert, AlertSeverity};
use crate::util::config::PagerDutyConfig;

/// PagerDuty Events API v2 client.
pub struct PagerDutyClient {
    /// Configuration.
    config: PagerDutyConfig,

    /// HTTP client for API requests.
    http: reqwest::Client,
}

impl PagerDutyClient {
    /// Create a new PagerDuty client.
    pub fn new(config: PagerDutyConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to create HTTP client");

        Self { config, http }
    }

    /// Send an alert to PagerDuty as a trigger event.
    pub async fn send_alert(&self, alert: &Alert) -> Result<()> {
        let severity = match alert.severity {
            AlertSeverity::Info => "info",
            AlertSeverity::Warning => "warning",
            AlertSeverity::High => "error",
            AlertSeverity::Critical => "critical",
        };

        let payload = json!({
            "routing_key": self.config.routing_key,
            "event_action": "trigger",
            "dedup_key": alert.dedup_key,
            "payload": {
                "summary": alert.summary,
                "severity": severity,
                "source": format!("sentinel-correlation-engine:{}",
                    alert.hostname.as_deref().unwrap_or("unknown")),
                "component": "GPU",
                "group": alert.hostname.as_deref().unwrap_or("unknown"),
                "class": "silent_data_corruption",
                "custom_details": {
                    "alert_id": alert.alert_id,
                    "gpu_uuid": alert.gpu_uuid,
                    "description": alert.description,
                    "metadata": alert.metadata,
                }
            }
        });

        let response = self
            .http
            .post(&self.config.api_url)
            .json(&payload)
            .send()
            .await
            .context("failed to send PagerDuty event")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            anyhow::bail!("PagerDuty API returned {}: {}", status, body);
        }

        debug!(
            dedup_key = %alert.dedup_key,
            "PagerDuty trigger event sent"
        );

        Ok(())
    }

    /// Resolve a PagerDuty incident by dedup key.
    pub async fn resolve(&self, dedup_key: &str) -> Result<()> {
        let payload = json!({
            "routing_key": self.config.routing_key,
            "event_action": "resolve",
            "dedup_key": dedup_key,
        });

        let response = self
            .http
            .post(&self.config.api_url)
            .json(&payload)
            .send()
            .await
            .context("failed to send PagerDuty resolve event")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            anyhow::bail!("PagerDuty API returned {}: {}", status, body);
        }

        debug!(dedup_key, "PagerDuty resolve event sent");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerduty_client_creation() {
        let config = PagerDutyConfig::default();
        let _client = PagerDutyClient::new(config);
    }
}
