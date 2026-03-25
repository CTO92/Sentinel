//! SENTINEL Audit Ledger — entry point.
//!
//! Reads configuration, connects to PostgreSQL, runs schema migrations, resumes
//! the hash chain from the last committed entry, and starts the gRPC server
//! together with background workers (periodic verification, partition
//! maintenance, metrics update).

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::{error, info};

use sentinel_audit_ledger::grpc;
use sentinel_audit_ledger::ledger::merkle_chain::ChainBuilder;
use sentinel_audit_ledger::ledger::pruning::{self, PruningPolicy};
use sentinel_audit_ledger::ledger::verification;
use sentinel_audit_ledger::storage::postgres;
use sentinel_audit_ledger::util::metrics;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Application configuration, loaded from environment or config file.
#[derive(Debug, Clone)]
struct AppConfig {
    /// PostgreSQL connection URL.
    database_url: String,
    /// Maximum database connections.
    max_db_connections: u32,
    /// gRPC listen address.
    grpc_addr: SocketAddr,
    /// Data retention in days.
    retention_days: u32,
    /// Chain verification interval in seconds.
    verification_interval_secs: u64,
}

impl AppConfig {
    fn from_env() -> Self {
        Self {
            database_url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| "postgres://sentinel:sentinel@localhost:5432/sentinel_audit".into()),
            max_db_connections: std::env::var("MAX_DB_CONNECTIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(20),
            grpc_addr: std::env::var("GRPC_ADDR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| "0.0.0.0:50052".parse().unwrap()),
            retention_days: std::env::var("RETENTION_DAYS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(pruning::DEFAULT_RETENTION_DAYS),
            verification_interval_secs: std::env::var("VERIFICATION_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3600),
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // Initialise structured logging.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,sqlx=warn".parse().unwrap()),
        )
        .json()
        .init();

    info!("SENTINEL Audit Ledger starting");

    let cfg = AppConfig::from_env();

    // Connect to PostgreSQL.
    let pool = postgres::create_pool(&cfg.database_url, cfg.max_db_connections)
        .await
        .context("creating database pool")?;

    // Run migrations.
    postgres::run_migrations(&pool).await.context("running migrations")?;

    // Resume chain state from the database.
    let chain_builder = resume_chain(&pool).await?;

    // Spawn background workers.
    let pool_bg = pool.clone();
    let verification_interval = Duration::from_secs(cfg.verification_interval_secs);
    tokio::spawn(async move {
        verification_worker(pool_bg, verification_interval).await;
    });

    let pool_prune = pool.clone();
    let pruning_policy = PruningPolicy {
        retention_days: cfg.retention_days,
    };
    tokio::spawn(async move {
        pruning_worker(pool_prune, pruning_policy).await;
    });

    let pool_metrics = pool.clone();
    tokio::spawn(async move {
        metrics_worker(pool_metrics).await;
    });

    // Start the gRPC server (blocks until shutdown).
    grpc::server::start(cfg.grpc_addr, pool, chain_builder).await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Chain resumption
// ---------------------------------------------------------------------------

async fn resume_chain(pool: &sqlx::PgPool) -> Result<ChainBuilder> {
    let last_entry = postgres::get_last_entry(pool).await?;
    let last_batch_seq = postgres::get_last_batch_sequence(pool).await?;

    match last_entry {
        Some(entry) => {
            info!(
                entry_id = entry.entry_id,
                batch_sequence = last_batch_seq,
                "Resuming chain from existing state"
            );
            Ok(ChainBuilder::resume(
                entry.entry_hash,
                entry.entry_id + 1,
                last_batch_seq + 1,
            ))
        }
        None => {
            info!("Starting fresh chain (no existing entries)");
            Ok(ChainBuilder::new())
        }
    }
}

// ---------------------------------------------------------------------------
// Background workers
// ---------------------------------------------------------------------------

/// Periodically verify the most recent portion of the chain.
async fn verification_worker(pool: sqlx::PgPool, interval: Duration) {
    loop {
        tokio::time::sleep(interval).await;

        info!("Running periodic chain verification");
        let query = postgres::AuditQuery {
            // Verify the last 24 hours of entries.
            start_time: Some(chrono::Utc::now() - chrono::Duration::hours(24)),
            ..Default::default()
        };

        match postgres::query_entries(&pool, &query).await {
            Ok(entries) => {
                let result = verification::verify_chain(&entries, None);
                if result.valid {
                    info!(
                        entries_checked = result.entries_checked,
                        "Chain verification passed"
                    );
                    metrics::CHAIN_VERIFICATION_RESULT
                        .with_label_values(&["valid"])
                        .inc();
                } else {
                    error!(
                        failures = result.failures,
                        "CHAIN VERIFICATION FAILED — possible tampering detected"
                    );
                    metrics::CHAIN_VERIFICATION_RESULT
                        .with_label_values(&["invalid"])
                        .inc();
                }
            }
            Err(e) => {
                error!(error = %e, "Failed to query entries for verification");
            }
        }
    }
}

/// Periodically prune old partitions and create future ones.
async fn pruning_worker(pool: sqlx::PgPool, policy: PruningPolicy) {
    // Run once per day.
    let interval = Duration::from_secs(86400);
    loop {
        tokio::time::sleep(interval).await;

        info!("Running partition maintenance");
        if let Err(e) = pruning::prune_old_partitions(&pool, &policy).await {
            error!(error = %e, "Partition pruning failed");
        }
        if let Err(e) = pruning::ensure_future_partitions(&pool, 4).await {
            error!(error = %e, "Future partition creation failed");
        }
    }
}

/// Periodically update the storage-size metric.
async fn metrics_worker(pool: sqlx::PgPool) {
    let interval = Duration::from_secs(300);
    loop {
        tokio::time::sleep(interval).await;

        match postgres::estimate_storage_bytes(&pool).await {
            Ok(bytes) => {
                metrics::AUDIT_STORAGE_BYTES.set(bytes);
            }
            Err(e) => {
                error!(error = %e, "Failed to estimate storage size");
            }
        }
    }
}
