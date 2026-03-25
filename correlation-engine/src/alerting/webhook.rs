//! Generic webhook integration for custom alert destinations.
//!
//! Sends JSON-formatted alert payloads to configurable HTTP endpoints.
//! Supports optional HMAC-SHA256 payload signing for webhook verification.

use anyhow::{Context, Result};
use sha2::Sha256;
use tracing::debug;

use crate::alerting::alert_manager::Alert;
use crate::util::config::WebhookConfig;

/// Generic webhook client.
pub struct WebhookClient {
    /// Configuration.
    config: WebhookConfig,

    /// HTTP client for webhook requests.
    http: reqwest::Client,
}

impl WebhookClient {
    /// Create a new webhook client.
    pub fn new(config: WebhookConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to create HTTP client");

        Self { config, http }
    }

    /// Get the webhook name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Send an alert to the webhook endpoint.
    pub async fn send_alert(&self, alert: &Alert) -> Result<()> {
        let payload = serde_json::to_string(alert)?;

        let mut request = self
            .http
            .post(&self.config.url)
            .header("Content-Type", "application/json");

        // Add configured headers.
        for (key, value) in &self.config.headers {
            request = request.header(key.as_str(), value.as_str());
        }

        // Add HMAC signature if secret is configured.
        if !self.config.secret.is_empty() {
            let signature = self.compute_hmac(&payload);
            request = request.header("X-Sentinel-Signature", format!("sha256={}", signature));
        }

        let response = request
            .body(payload)
            .send()
            .await
            .context("failed to send webhook")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            anyhow::bail!(
                "Webhook '{}' returned {}: {}",
                self.config.name,
                status,
                body
            );
        }

        debug!(
            name = %self.config.name,
            alert_id = %alert.alert_id,
            "Webhook alert sent"
        );

        Ok(())
    }

    /// Compute HMAC-SHA256 of the payload using the configured secret.
    fn compute_hmac(&self, payload: &str) -> String {
        use sha2::Digest;

        let mut hasher = Sha256::new();
        hasher.update(self.config.secret.as_bytes());
        hasher.update(payload.as_bytes());
        let result = hasher.finalize();

        hex::encode(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webhook_client_creation() {
        let config = WebhookConfig {
            name: "test".to_string(),
            url: "http://localhost:8080/webhook".to_string(),
            enabled: true,
            secret: "my-secret".to_string(),
            headers: std::collections::HashMap::new(),
        };
        let client = WebhookClient::new(config);
        assert_eq!(client.name(), "test");
    }

    #[test]
    fn test_hmac_computation() {
        let config = WebhookConfig {
            name: "test".to_string(),
            url: "http://localhost:8080".to_string(),
            enabled: true,
            secret: "secret123".to_string(),
            headers: std::collections::HashMap::new(),
        };
        let client = WebhookClient::new(config);
        let sig = client.compute_hmac("test payload");
        assert!(!sig.is_empty());
        // Same input should produce same signature.
        assert_eq!(sig, client.compute_hmac("test payload"));
    }
}
