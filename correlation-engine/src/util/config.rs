//! Application configuration loaded from YAML files and environment variables.
//!
//! The configuration hierarchy (lowest to highest priority):
//! 1. Built-in defaults
//! 2. YAML config file (`SENTINEL_CONFIG` env var or `config/correlation-engine.yaml`)
//! 3. Environment variable overrides (`SENTINEL_` prefix)

use serde::Deserialize;
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during configuration loading.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to load configuration: {0}")]
    LoadError(#[from] config::ConfigError),

    #[error("invalid configuration value: {field}: {reason}")]
    ValidationError { field: String, reason: String },
}

/// Top-level configuration for the Correlation Engine.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// gRPC server settings.
    pub grpc: GrpcConfig,

    /// REST API server settings (for the dashboard).
    pub rest: RestConfig,

    /// Bayesian attribution model parameters.
    pub bayesian: BayesianConfig,

    /// Quarantine state machine thresholds.
    pub quarantine: QuarantineConfig,

    /// Temporal correlation window settings.
    pub temporal_window: TemporalWindowConfig,

    /// TMR (Triple Modular Redundancy) scheduling settings.
    pub tmr: TmrConfig,

    /// Trust graph settings.
    pub trust: TrustConfig,

    /// Redis connection settings.
    pub redis: RedisConfig,

    /// Alerting configuration.
    pub alerting: AlertingConfig,

    /// Metrics / Prometheus settings.
    pub metrics: MetricsConfig,

    /// Logging settings.
    pub logging: LoggingConfig,

    /// Event store settings.
    pub event_store: EventStoreConfig,

    /// Audit ledger client settings.
    pub audit: AuditConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            grpc: GrpcConfig::default(),
            rest: RestConfig::default(),
            bayesian: BayesianConfig::default(),
            quarantine: QuarantineConfig::default(),
            temporal_window: TemporalWindowConfig::default(),
            tmr: TmrConfig::default(),
            trust: TrustConfig::default(),
            redis: RedisConfig::default(),
            alerting: AlertingConfig::default(),
            metrics: MetricsConfig::default(),
            logging: LoggingConfig::default(),
            event_store: EventStoreConfig::default(),
            audit: AuditConfig::default(),
        }
    }
}

/// gRPC server configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GrpcConfig {
    /// Bind address for the gRPC server.
    pub bind_addr: String,

    /// Port for the gRPC server.
    pub port: u16,

    /// Maximum concurrent gRPC connections.
    pub max_connections: usize,

    /// Request timeout in seconds.
    pub request_timeout_secs: u64,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0".to_string(),
            port: 50051,
            max_connections: 1024,
            request_timeout_secs: 30,
        }
    }
}

/// REST API (axum) server configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RestConfig {
    /// Bind address for the REST server.
    pub bind_addr: String,

    /// Port for the REST server.
    pub port: u16,
}

impl Default for RestConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0".to_string(),
            port: 8080,
        }
    }
}

/// Bayesian attribution model configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BayesianConfig {
    /// Prior alpha for the Beta distribution (successful probes).
    pub prior_alpha: f64,

    /// Prior beta for the Beta distribution (failed probes).
    pub prior_beta: f64,

    /// Weight contributed by a probe failure to the beta parameter.
    pub probe_failure_weight: f64,

    /// Weight contributed by an anomaly event to the beta parameter.
    pub anomaly_weight: f64,

    /// Reliability threshold below which probing frequency is increased.
    pub increased_probing_threshold: f64,

    /// Reliability threshold below which GPU enters SUSPECT state.
    pub suspect_threshold: f64,

    /// Reliability threshold below which GPU enters QUARANTINED state.
    pub quarantine_threshold: f64,

    /// Reliability threshold below which GPU is CONDEMNED.
    pub condemned_threshold: f64,
}

impl Default for BayesianConfig {
    fn default() -> Self {
        Self {
            prior_alpha: 1000.0,
            prior_beta: 1.0,
            probe_failure_weight: 1.0,
            anomaly_weight: 0.3,
            increased_probing_threshold: 0.999,
            suspect_threshold: 0.995,
            quarantine_threshold: 0.99,
            condemned_threshold: 0.95,
        }
    }
}

/// Quarantine state machine configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct QuarantineConfig {
    /// Number of consecutive successful deep-test probes to reinstate a GPU.
    pub reinstatement_pass_count: u64,

    /// Maximum time in seconds a GPU can stay in QUARANTINED before escalation.
    pub quarantine_timeout_secs: u64,

    /// Maximum time in seconds for deep-test phase.
    pub deep_test_timeout_secs: u64,

    /// Whether condemn actions require human approval.
    pub condemn_requires_approval: bool,
}

impl Default for QuarantineConfig {
    fn default() -> Self {
        Self {
            reinstatement_pass_count: 1000,
            quarantine_timeout_secs: 86400,      // 24 hours
            deep_test_timeout_secs: 14400,        // 4 hours
            condemn_requires_approval: true,
        }
    }
}

/// Temporal correlation window configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TemporalWindowConfig {
    /// Window duration in seconds.
    pub window_duration_secs: u64,

    /// How often to expire old events from the window (seconds).
    pub cleanup_interval_secs: u64,
}

impl Default for TemporalWindowConfig {
    fn default() -> Self {
        Self {
            window_duration_secs: 300, // 5 minutes
            cleanup_interval_secs: 10,
        }
    }
}

/// TMR scheduling configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TmrConfig {
    /// Interval between TMR canary runs (seconds).
    pub interval_secs: u64,

    /// Number of GPU triples to schedule per interval.
    pub triples_per_interval: usize,

    /// Timeout for each TMR computation (milliseconds).
    pub timeout_ms: u32,
}

impl Default for TmrConfig {
    fn default() -> Self {
        Self {
            interval_secs: 60,
            triples_per_interval: 10,
            timeout_ms: 5000,
        }
    }
}

/// Trust graph configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TrustConfig {
    /// Minimum number of comparisons before a trust edge is considered valid.
    pub min_comparisons: u64,

    /// Decay factor for old trust edges (0.0 to 1.0, applied per hour).
    pub decay_factor: f64,
}

impl Default for TrustConfig {
    fn default() -> Self {
        Self {
            min_comparisons: 10,
            decay_factor: 0.999,
        }
    }
}

/// Redis connection configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RedisConfig {
    /// Redis connection URL.
    pub url: String,

    /// Key prefix for all Sentinel keys.
    pub key_prefix: String,

    /// Connection pool size.
    pub pool_size: usize,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".to_string(),
            key_prefix: "sentinel:".to_string(),
            pool_size: 8,
        }
    }
}

/// Alerting configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AlertingConfig {
    /// Whether alerting is enabled.
    pub enabled: bool,

    /// Rate limit: minimum seconds between alerts of the same type for the same GPU.
    pub rate_limit_secs: u64,

    /// Maximum alerts per minute across all channels.
    pub max_alerts_per_minute: u32,

    /// PagerDuty integration settings.
    pub pagerduty: PagerDutyConfig,

    /// Slack integration settings.
    pub slack: SlackConfig,

    /// Generic webhook settings.
    pub webhooks: Vec<WebhookConfig>,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rate_limit_secs: 300,
            max_alerts_per_minute: 60,
            pagerduty: PagerDutyConfig::default(),
            slack: SlackConfig::default(),
            webhooks: Vec::new(),
        }
    }
}

/// PagerDuty configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PagerDutyConfig {
    /// Whether PagerDuty integration is enabled.
    pub enabled: bool,

    /// PagerDuty Events API v2 routing key.
    pub routing_key: String,

    /// PagerDuty API base URL.
    pub api_url: String,
}

impl Default for PagerDutyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            routing_key: String::new(),
            api_url: "https://events.pagerduty.com/v2/enqueue".to_string(),
        }
    }
}

/// Slack configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SlackConfig {
    /// Whether Slack integration is enabled.
    pub enabled: bool,

    /// Slack incoming webhook URL.
    pub webhook_url: String,

    /// Channel override (if different from webhook default).
    pub channel: String,
}

impl Default for SlackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            webhook_url: String::new(),
            channel: String::new(),
        }
    }
}

/// Generic webhook configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WebhookConfig {
    /// Human-readable name for this webhook.
    pub name: String,

    /// Webhook URL.
    pub url: String,

    /// Whether this webhook is enabled.
    pub enabled: bool,

    /// Optional HMAC secret for signing payloads.
    pub secret: String,

    /// HTTP headers to include.
    pub headers: std::collections::HashMap<String, String>,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            url: String::new(),
            enabled: false,
            secret: String::new(),
            headers: std::collections::HashMap::new(),
        }
    }
}

/// Prometheus metrics configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MetricsConfig {
    /// Bind address for the Prometheus scrape endpoint.
    pub bind_addr: String,

    /// Port for the metrics endpoint.
    pub port: u16,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0".to_string(),
            port: 9090,
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level filter (e.g., "info", "debug", "trace").
    pub level: String,

    /// Whether to use JSON structured logging.
    pub json: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json: false,
        }
    }
}

/// Event store configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EventStoreConfig {
    /// Maximum events to retain in the in-memory ring buffer per GPU.
    pub ring_buffer_size: usize,

    /// Batch flush interval in milliseconds.
    pub flush_interval_ms: u64,

    /// Maximum batch size before forced flush.
    pub max_batch_size: usize,
}

impl Default for EventStoreConfig {
    fn default() -> Self {
        Self {
            ring_buffer_size: 10_000,
            flush_interval_ms: 100,
            max_batch_size: 500,
        }
    }
}

/// Audit ledger client configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AuditConfig {
    /// gRPC endpoint for the Audit Ledger service.
    pub endpoint: String,

    /// Connection timeout in seconds.
    pub connect_timeout_secs: u64,

    /// Request timeout in seconds.
    pub request_timeout_secs: u64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:50052".to_string(),
            connect_timeout_secs: 5,
            request_timeout_secs: 10,
        }
    }
}

/// Load configuration from the default file path, environment variables, and
/// optional overrides.
///
/// The config file path is determined by:
/// 1. `SENTINEL_CONFIG` environment variable
/// 2. Default: `config/correlation-engine.yaml` relative to CWD
pub fn load_config(config_path: Option<PathBuf>) -> Result<AppConfig, ConfigError> {
    let default_path = PathBuf::from("config/correlation-engine.yaml");
    let path = config_path
        .or_else(|| std::env::var("SENTINEL_CONFIG").ok().map(PathBuf::from))
        .unwrap_or(default_path);

    let mut builder = config::Config::builder()
        // Start with compiled-in defaults.
        .set_default("grpc.port", 50051)?
        .set_default("rest.port", 8080)?;

    // Layer the YAML file if it exists.
    if path.exists() {
        builder = builder.add_source(config::File::from(path));
    }

    // Layer environment variable overrides with SENTINEL_ prefix.
    builder = builder.add_source(
        config::Environment::with_prefix("SENTINEL")
            .separator("__")
            .try_parsing(true),
    );

    let cfg: AppConfig = builder.build()?.try_deserialize().unwrap_or_default();
    validate_config(&cfg)?;
    Ok(cfg)
}

/// Validate configuration values for correctness.
fn validate_config(cfg: &AppConfig) -> Result<(), ConfigError> {
    if cfg.bayesian.prior_alpha <= 0.0 {
        return Err(ConfigError::ValidationError {
            field: "bayesian.prior_alpha".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    if cfg.bayesian.prior_beta <= 0.0 {
        return Err(ConfigError::ValidationError {
            field: "bayesian.prior_beta".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    if cfg.bayesian.suspect_threshold >= cfg.bayesian.increased_probing_threshold {
        return Err(ConfigError::ValidationError {
            field: "bayesian.suspect_threshold".to_string(),
            reason: "must be less than increased_probing_threshold".to_string(),
        });
    }
    if cfg.bayesian.quarantine_threshold >= cfg.bayesian.suspect_threshold {
        return Err(ConfigError::ValidationError {
            field: "bayesian.quarantine_threshold".to_string(),
            reason: "must be less than suspect_threshold".to_string(),
        });
    }
    if cfg.bayesian.condemned_threshold >= cfg.bayesian.quarantine_threshold {
        return Err(ConfigError::ValidationError {
            field: "bayesian.condemned_threshold".to_string(),
            reason: "must be less than quarantine_threshold".to_string(),
        });
    }
    if cfg.temporal_window.window_duration_secs == 0 {
        return Err(ConfigError::ValidationError {
            field: "temporal_window.window_duration_secs".to_string(),
            reason: "must be non-zero".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let cfg = AppConfig::default();
        assert!(validate_config(&cfg).is_ok());
    }

    #[test]
    fn test_invalid_prior_alpha() {
        let mut cfg = AppConfig::default();
        cfg.bayesian.prior_alpha = -1.0;
        assert!(validate_config(&cfg).is_err());
    }

    #[test]
    fn test_invalid_threshold_ordering() {
        let mut cfg = AppConfig::default();
        cfg.bayesian.suspect_threshold = 0.9999;
        cfg.bayesian.increased_probing_threshold = 0.999;
        assert!(validate_config(&cfg).is_err());
    }
}
