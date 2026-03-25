//! Slack webhook integration for alert notifications.
//!
//! Sends formatted messages to Slack channels using incoming webhooks.
//! Messages include alert details, GPU identification, and direct links
//! to the SENTINEL dashboard.

use anyhow::{Context, Result};
use serde_json::json;
use tracing::debug;

use crate::alerting::alert_manager::{Alert, AlertSeverity};
use crate::util::config::SlackConfig;

/// Slack incoming webhook client.
pub struct SlackClient {
    /// Configuration.
    config: SlackConfig,

    /// HTTP client for webhook requests.
    http: reqwest::Client,
}

impl SlackClient {
    /// Create a new Slack client.
    pub fn new(config: SlackConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to create HTTP client");

        Self { config, http }
    }

    /// Send an alert to Slack.
    pub async fn send_alert(&self, alert: &Alert) -> Result<()> {
        let color = match alert.severity {
            AlertSeverity::Info => "#2196F3",      // Blue
            AlertSeverity::Warning => "#FF9800",    // Orange
            AlertSeverity::High => "#F44336",       // Red
            AlertSeverity::Critical => "#9C27B0",   // Purple
        };

        let severity_emoji = match alert.severity {
            AlertSeverity::Info => "info",
            AlertSeverity::Warning => "warning",
            AlertSeverity::High => "high",
            AlertSeverity::Critical => "critical",
        };

        let mut fields = vec![
            json!({
                "title": "Severity",
                "value": severity_emoji,
                "short": true
            }),
            json!({
                "title": "Source",
                "value": alert.source,
                "short": true
            }),
        ];

        if let Some(ref gpu) = alert.gpu_uuid {
            fields.push(json!({
                "title": "GPU UUID",
                "value": format!("`{}`", gpu),
                "short": false
            }));
        }

        if let Some(ref host) = alert.hostname {
            fields.push(json!({
                "title": "Hostname",
                "value": host,
                "short": true
            }));
        }

        let mut payload = json!({
            "attachments": [{
                "color": color,
                "title": format!("SENTINEL Alert: {}", alert.summary),
                "text": alert.description,
                "fields": fields,
                "footer": "SENTINEL Correlation Engine",
                "ts": alert.timestamp.timestamp()
            }]
        });

        if !self.config.channel.is_empty() {
            payload["channel"] = json!(self.config.channel);
        }

        let response = self
            .http
            .post(&self.config.webhook_url)
            .json(&payload)
            .send()
            .await
            .context("failed to send Slack webhook")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            anyhow::bail!("Slack webhook returned {}: {}", status, body);
        }

        debug!(
            alert_id = %alert.alert_id,
            "Slack alert sent"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slack_client_creation() {
        let config = SlackConfig::default();
        let _client = SlackClient::new(config);
    }
}
