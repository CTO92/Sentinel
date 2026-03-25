//! PostgreSQL storage backend for the SENTINEL Audit Ledger.
//!
//! Handles connection pooling, schema migrations, batch inserts, and queries.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use sqlx::postgres::{PgPoolOptions, PgRow};
use sqlx::{PgPool, Row};
use tracing::info;

use crate::ledger::entry::{AuditEntry, AuditEntryType};
use crate::ledger::merkle_chain::CommittedBatch;

// ---------------------------------------------------------------------------
// Pool construction
// ---------------------------------------------------------------------------

/// Build a PostgreSQL connection pool.
pub async fn create_pool(database_url: &str, max_connections: u32) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(max_connections)
        .acquire_timeout(std::time::Duration::from_secs(10))
        .idle_timeout(std::time::Duration::from_secs(600))
        .connect(database_url)
        .await
        .context("connecting to PostgreSQL")?;

    info!("PostgreSQL connection pool established");
    Ok(pool)
}

// ---------------------------------------------------------------------------
// Migrations
// ---------------------------------------------------------------------------

/// Run embedded SQL migrations in order.
pub async fn run_migrations(pool: &PgPool) -> Result<()> {
    let migrations: &[(&str, &str)] = &[
        ("001_initial", include_str!("migrations/001_initial.sql")),
        ("002_indices", include_str!("migrations/002_indices.sql")),
        (
            "003_partitioning",
            include_str!("migrations/003_partitioning.sql"),
        ),
    ];

    for (name, sql) in migrations {
        info!(migration = name, "Applying migration");
        sqlx::raw_sql(sql)
            .execute(pool)
            .await
            .with_context(|| format!("running migration {name}"))?;
    }

    info!("All migrations applied successfully");
    Ok(())
}

// ---------------------------------------------------------------------------
// Batch insert
// ---------------------------------------------------------------------------

/// Write a committed batch to PostgreSQL in a single transaction.
pub async fn insert_batch(pool: &PgPool, batch: &CommittedBatch) -> Result<()> {
    let mut tx = pool.begin().await.context("begin transaction")?;

    for entry in &batch.entries {
        sqlx::query(
            "INSERT INTO audit_entries \
             (entry_id, entry_type, timestamp, gpu_uuid, sm_id, data, \
              previous_hash, entry_hash, merkle_root, batch_sequence) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
        )
        .bind(entry.entry_id as i64)
        .bind(entry.entry_type.as_i16())
        .bind(entry.timestamp)
        .bind(&entry.gpu_uuid)
        .bind(entry.sm_id)
        .bind(&entry.data)
        .bind(&entry.previous_hash[..])
        .bind(&entry.entry_hash[..])
        .bind(entry.merkle_root.as_ref().map(|h| &h[..]))
        .bind(entry.batch_sequence as i64)
        .execute(&mut *tx)
        .await
        .with_context(|| format!("inserting entry {}", entry.entry_id))?;
    }

    // Record the Merkle root for this batch.
    let first_id = batch.entries.first().map(|e| e.entry_id as i64).unwrap_or(0);
    let last_id = batch.entries.last().map(|e| e.entry_id as i64).unwrap_or(0);

    sqlx::query(
        "INSERT INTO merkle_roots \
         (batch_sequence, merkle_root, entry_count, first_entry_id, last_entry_id) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(batch.batch_sequence as i64)
    .bind(&batch.merkle_root[..])
    .bind(batch.entries.len() as i32)
    .bind(first_id)
    .bind(last_id)
    .execute(&mut *tx)
    .await
    .context("inserting merkle root")?;

    tx.commit().await.context("commit transaction")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// Filter for querying audit entries.
#[derive(Debug, Default, Clone)]
pub struct AuditQuery {
    pub gpu_uuid: Option<String>,
    pub entry_type: Option<AuditEntryType>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// Query audit entries with optional filters.
pub async fn query_entries(pool: &PgPool, q: &AuditQuery) -> Result<Vec<AuditEntry>> {
    // Build a dynamic query. We use a simple approach with conditional clauses.
    let mut sql = String::from(
        "SELECT entry_id, entry_type, timestamp, gpu_uuid, sm_id, \
         data, previous_hash, entry_hash, merkle_root, batch_sequence \
         FROM audit_entries WHERE 1=1",
    );
    let mut bind_idx: u32 = 0;

    if q.gpu_uuid.is_some() {
        bind_idx += 1;
        sql.push_str(&format!(" AND gpu_uuid = ${bind_idx}"));
    }
    if q.entry_type.is_some() {
        bind_idx += 1;
        sql.push_str(&format!(" AND entry_type = ${bind_idx}"));
    }
    if q.start_time.is_some() {
        bind_idx += 1;
        sql.push_str(&format!(" AND timestamp >= ${bind_idx}"));
    }
    if q.end_time.is_some() {
        bind_idx += 1;
        sql.push_str(&format!(" AND timestamp < ${bind_idx}"));
    }

    sql.push_str(" ORDER BY entry_id ASC");

    if q.limit.is_some() {
        bind_idx += 1;
        sql.push_str(&format!(" LIMIT ${bind_idx}"));
    }
    if q.offset.is_some() {
        bind_idx += 1;
        sql.push_str(&format!(" OFFSET ${bind_idx}"));
    }

    // We cannot use compile-time checked queries with dynamic SQL, so we use
    // sqlx::query() and bind manually.
    let mut query = sqlx::query(&sql);

    if let Some(ref gpu) = q.gpu_uuid {
        query = query.bind(gpu);
    }
    if let Some(et) = q.entry_type {
        query = query.bind(et.as_i16());
    }
    if let Some(st) = q.start_time {
        query = query.bind(st);
    }
    if let Some(et) = q.end_time {
        query = query.bind(et);
    }
    if let Some(lim) = q.limit {
        query = query.bind(lim);
    }
    if let Some(off) = q.offset {
        query = query.bind(off);
    }

    let rows = query.fetch_all(pool).await.context("query_entries")?;
    rows.iter().map(row_to_entry).collect()
}

/// Fetch entries for a given batch sequence number.
pub async fn get_batch_entries(pool: &PgPool, batch_sequence: u64) -> Result<Vec<AuditEntry>> {
    let rows = sqlx::query(
        "SELECT entry_id, entry_type, timestamp, gpu_uuid, sm_id, \
         data, previous_hash, entry_hash, merkle_root, batch_sequence \
         FROM audit_entries WHERE batch_sequence = $1 ORDER BY entry_id ASC",
    )
    .bind(batch_sequence as i64)
    .fetch_all(pool)
    .await
    .context("get_batch_entries")?;

    rows.iter().map(row_to_entry).collect()
}

/// Fetch the Merkle root for a batch.
pub async fn get_merkle_root(pool: &PgPool, batch_sequence: u64) -> Result<Option<Vec<u8>>> {
    let row: Option<(Vec<u8>,)> = sqlx::query_as(
        "SELECT merkle_root FROM merkle_roots WHERE batch_sequence = $1",
    )
    .bind(batch_sequence as i64)
    .fetch_optional(pool)
    .await
    .context("get_merkle_root")?;

    Ok(row.map(|(r,)| r))
}

/// Get the last entry in the chain (for resuming after restart).
pub async fn get_last_entry(pool: &PgPool) -> Result<Option<AuditEntry>> {
    let row = sqlx::query(
        "SELECT entry_id, entry_type, timestamp, gpu_uuid, sm_id, \
         data, previous_hash, entry_hash, merkle_root, batch_sequence \
         FROM audit_entries ORDER BY entry_id DESC LIMIT 1",
    )
    .fetch_optional(pool)
    .await
    .context("get_last_entry")?;

    match row {
        Some(ref r) => Ok(Some(row_to_entry(r)?)),
        None => Ok(None),
    }
}

/// Get the latest batch sequence number.
pub async fn get_last_batch_sequence(pool: &PgPool) -> Result<u64> {
    let row: Option<(i64,)> =
        sqlx::query_as("SELECT COALESCE(MAX(batch_sequence), 0) FROM merkle_roots")
            .fetch_optional(pool)
            .await
            .context("get_last_batch_sequence")?;

    Ok(row.map(|(v,)| v as u64).unwrap_or(0))
}

/// Estimate total storage used by the audit_entries table (bytes).
pub async fn estimate_storage_bytes(pool: &PgPool) -> Result<i64> {
    let row: Option<(i64,)> = sqlx::query_as(
        "SELECT pg_total_relation_size('audit_entries'::regclass)",
    )
    .fetch_optional(pool)
    .await
    .context("estimate_storage_bytes")?;

    Ok(row.map(|(v,)| v).unwrap_or(0))
}

/// Record an external chain anchor (e.g. from a timestamping service).
pub async fn insert_chain_anchor(
    pool: &PgPool,
    batch_sequence: u64,
    merkle_root: &[u8],
    external_timestamp: Option<&str>,
    anchor_service: Option<&str>,
) -> Result<()> {
    sqlx::query(
        "INSERT INTO chain_anchors \
         (batch_sequence, merkle_root, external_timestamp, anchor_service) \
         VALUES ($1, $2, $3, $4)",
    )
    .bind(batch_sequence as i64)
    .bind(merkle_root)
    .bind(external_timestamp)
    .bind(anchor_service)
    .execute(pool)
    .await
    .context("insert_chain_anchor")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Row mapping
// ---------------------------------------------------------------------------

fn row_to_entry(row: &PgRow) -> Result<AuditEntry> {
    let entry_id: i64 = row.try_get("entry_id")?;
    let entry_type_raw: i16 = row.try_get("entry_type")?;
    let timestamp: DateTime<Utc> = row.try_get("timestamp")?;
    let gpu_uuid: Option<String> = row.try_get("gpu_uuid")?;
    let sm_id: Option<i32> = row.try_get("sm_id")?;
    let data: Vec<u8> = row.try_get("data")?;
    let previous_hash_bytes: Vec<u8> = row.try_get("previous_hash")?;
    let entry_hash_bytes: Vec<u8> = row.try_get("entry_hash")?;
    let merkle_root_bytes: Option<Vec<u8>> = row.try_get("merkle_root")?;
    let batch_sequence: i64 = row.try_get("batch_sequence")?;

    let entry_type = AuditEntryType::from_i16(entry_type_raw)
        .ok_or_else(|| anyhow::anyhow!("unknown entry_type: {entry_type_raw}"))?;

    let mut previous_hash = [0u8; 32];
    if previous_hash_bytes.len() == 32 {
        previous_hash.copy_from_slice(&previous_hash_bytes);
    }

    let mut entry_hash = [0u8; 32];
    if entry_hash_bytes.len() == 32 {
        entry_hash.copy_from_slice(&entry_hash_bytes);
    }

    let merkle_root = merkle_root_bytes.and_then(|b| {
        if b.len() == 32 {
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&b);
            Some(arr)
        } else {
            None
        }
    });

    Ok(AuditEntry {
        entry_id: entry_id as u64,
        entry_type,
        timestamp,
        gpu_uuid,
        sm_id,
        data,
        previous_hash,
        entry_hash,
        merkle_root,
        batch_sequence: batch_sequence as u64,
    })
}
