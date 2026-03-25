//! Data retention and partition pruning for the audit ledger.
//!
//! The ledger retains data for a configurable period (default 7 years / 2555
//! days for SOC-2 / ISO 27001 compliance).  Pruning drops **monthly
//! partitions** that fall outside the retention window while preserving the
//! Merkle roots so that the chain of trust can still be verified even after
//! the underlying entries are gone.

use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use sqlx::PgPool;
use tracing::info;

/// Default retention period in days (7 years).
pub const DEFAULT_RETENTION_DAYS: u32 = 2555;

/// Configuration for the pruning policy.
#[derive(Debug, Clone)]
pub struct PruningPolicy {
    /// Number of days to retain audit entries.
    pub retention_days: u32,
}

impl Default for PruningPolicy {
    fn default() -> Self {
        Self {
            retention_days: DEFAULT_RETENTION_DAYS,
        }
    }
}

impl PruningPolicy {
    /// Compute the cutoff date: partitions whose month-end is strictly before
    /// this date are eligible for pruning.
    pub fn cutoff_date(&self) -> DateTime<Utc> {
        Utc::now() - chrono::Duration::days(self.retention_days as i64)
    }
}

/// Drop monthly partitions older than the retention policy.
///
/// Returns the list of partition names that were dropped.
pub async fn prune_old_partitions(pool: &PgPool, policy: &PruningPolicy) -> Result<Vec<String>> {
    let cutoff = policy.cutoff_date();
    info!(
        cutoff = %cutoff,
        retention_days = policy.retention_days,
        "Starting partition pruning"
    );

    // Discover partition tables whose name encodes a year_month earlier than
    // the cutoff.
    let partitions: Vec<(String,)> = sqlx::query_as(
        "SELECT tablename::text FROM pg_tables \
         WHERE schemaname = 'public' AND tablename LIKE 'audit_entries_%' \
         ORDER BY tablename",
    )
    .fetch_all(pool)
    .await
    .context("listing partitions")?;

    let mut dropped = Vec::new();

    for (name,) in &partitions {
        if let Some(partition_date) = parse_partition_month(name) {
            // The partition covers [partition_date, partition_date + 1 month).
            // We only drop if the *entire* month is before the cutoff.
            let end_of_month = add_one_month(partition_date);
            if end_of_month <= cutoff.date_naive() {
                // Before dropping, ensure the Merkle roots for this partition's
                // batches are preserved (they live in a separate table so
                // dropping the partition is sufficient).
                info!(partition = %name, "Dropping expired partition");
                let drop_sql = format!("DROP TABLE IF EXISTS {name}");
                sqlx::query(&drop_sql)
                    .execute(pool)
                    .await
                    .with_context(|| format!("dropping partition {name}"))?;
                dropped.push(name.clone());
            }
        }
    }

    if dropped.is_empty() {
        info!("No partitions eligible for pruning");
    } else {
        info!(count = dropped.len(), "Pruning complete");
    }

    // Record the pruning action itself as an audit entry (via direct insert so
    // we don't depend on the ingest pipeline being up).
    record_pruning_event(pool, &dropped).await?;

    Ok(dropped)
}

/// Ensure future monthly partitions exist for the next `months_ahead` months.
pub async fn ensure_future_partitions(pool: &PgPool, months_ahead: u32) -> Result<()> {
    for m in 0..months_ahead {
        let target = Utc::now().date_naive() + chrono::Duration::days(m as i64 * 30);
        let target_str = target.format("%Y-%m-%d").to_string();
        sqlx::query("SELECT create_monthly_partition($1::date)")
            .bind(&target_str)
            .execute(pool)
            .await
            .with_context(|| format!("creating partition for {target_str}"))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse a partition table name like `audit_entries_2025_03` into a `NaiveDate`
/// representing the first day of that month.
fn parse_partition_month(name: &str) -> Option<NaiveDate> {
    // Expected format: audit_entries_YYYY_MM
    let suffix = name.strip_prefix("audit_entries_")?;
    let parts: Vec<&str> = suffix.split('_').collect();
    if parts.len() != 2 {
        return None;
    }
    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    NaiveDate::from_ymd_opt(year, month, 1)
}

/// Add one calendar month to a `NaiveDate` at day 1.
fn add_one_month(date: NaiveDate) -> NaiveDate {
    if date.month() == 12 {
        NaiveDate::from_ymd_opt(date.year() + 1, 1, 1).unwrap()
    } else {
        NaiveDate::from_ymd_opt(date.year(), date.month() + 1, 1).unwrap()
    }
}

/// Insert a SystemEvent entry recording that pruning occurred.
async fn record_pruning_event(pool: &PgPool, dropped: &[String]) -> Result<()> {
    if dropped.is_empty() {
        return Ok(());
    }

    let payload = serde_json::json!({
        "action": "partition_prune",
        "dropped_partitions": dropped,
        "timestamp": Utc::now().to_rfc3339(),
    });
    let data = serde_json::to_vec(&payload)?;

    // We compute a hash but don't chain it into the main chain — this is an
    // out-of-band audit record.  A production deployment may choose to route
    // this through the normal ingest pipeline instead.
    let hash = crate::util::crypto::sha256(&data);

    sqlx::query(
        "INSERT INTO audit_entries \
         (entry_type, timestamp, data, previous_hash, entry_hash, batch_sequence) \
         VALUES (5, NOW(), $1, $2, $3, 0)",
    )
    .bind(&data)
    .bind(&hash[..]) // use own hash as placeholder for previous
    .bind(&hash[..])
    .execute(pool)
    .await
    .context("recording pruning event")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_partition_month() {
        let d = parse_partition_month("audit_entries_2025_03").unwrap();
        assert_eq!(d, NaiveDate::from_ymd_opt(2025, 3, 1).unwrap());
        assert!(parse_partition_month("other_table").is_none());
        assert!(parse_partition_month("audit_entries_bad").is_none());
    }

    #[test]
    fn test_add_one_month() {
        let jan = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        assert_eq!(
            add_one_month(jan),
            NaiveDate::from_ymd_opt(2025, 2, 1).unwrap()
        );
        let dec = NaiveDate::from_ymd_opt(2025, 12, 1).unwrap();
        assert_eq!(
            add_one_month(dec),
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
        );
    }

    #[test]
    fn test_cutoff_date_reasonable() {
        let policy = PruningPolicy::default();
        let cutoff = policy.cutoff_date();
        // Cutoff should be roughly 7 years in the past.
        let diff = Utc::now() - cutoff;
        assert!((diff.num_days() - 2555).abs() < 2);
    }
}
