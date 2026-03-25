//! Chain and entry verification for tamper detection.
//!
//! Provides functions to verify the integrity of the hash chain at various
//! granularities: single entry, batch (Merkle tree), arbitrary range, and the
//! entire chain.

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::ledger::entry::AuditEntry;
use crate::util::crypto::{self, Hash256, ZERO_HASH};

// ---------------------------------------------------------------------------
// Verification result types
// ---------------------------------------------------------------------------

/// Outcome of a verification operation.
#[derive(Debug, Clone, Serialize)]
pub struct VerificationResult {
    /// Whether the verification passed.
    pub valid: bool,
    /// Number of entries checked.
    pub entries_checked: u64,
    /// Number of entries that failed verification.
    pub failures: u64,
    /// Details of each failure (if any).
    pub failure_details: Vec<VerificationFailure>,
    /// Wall-clock time spent on verification.
    pub elapsed_ms: u64,
}

/// A single verification failure.
#[derive(Debug, Clone, Serialize)]
pub struct VerificationFailure {
    pub entry_id: u64,
    pub failure_type: FailureType,
    pub message: String,
}

/// Type of integrity failure detected.
#[derive(Debug, Clone, Serialize)]
pub enum FailureType {
    /// The entry's own hash does not match `SHA-256(data || previous_hash)`.
    HashMismatch,
    /// The entry's `previous_hash` field does not match the preceding entry's hash.
    ChainBreak,
    /// The Merkle root stored at a batch boundary does not match the recomputed value.
    MerkleRootMismatch,
}

// ---------------------------------------------------------------------------
// Verification functions
// ---------------------------------------------------------------------------

/// Verify the hash chain over a slice of **consecutive** entries.
///
/// The entries must be ordered by `entry_id` and contiguous.  If
/// `expected_previous_hash` is `None` and the first entry has `entry_id == 0`
/// or `entry_id == 1`, [`ZERO_HASH`] is assumed; otherwise the caller must
/// supply the predecessor's hash.
pub fn verify_chain(
    entries: &[AuditEntry],
    expected_previous_hash: Option<Hash256>,
) -> VerificationResult {
    let start = std::time::Instant::now();
    let mut failures = Vec::new();

    if entries.is_empty() {
        return VerificationResult {
            valid: true,
            entries_checked: 0,
            failures: 0,
            failure_details: vec![],
            elapsed_ms: 0,
        };
    }

    let mut prev_hash = expected_previous_hash.unwrap_or_else(|| {
        if entries[0].entry_id <= 1 {
            ZERO_HASH
        } else {
            // We don't know the predecessor; trust the first entry's claim.
            entries[0].previous_hash
        }
    });

    for entry in entries {
        // 1. Check chain link.
        if !crypto::constant_time_eq(&entry.previous_hash, &prev_hash) {
            failures.push(VerificationFailure {
                entry_id: entry.entry_id,
                failure_type: FailureType::ChainBreak,
                message: format!(
                    "previous_hash mismatch at entry {}: expected {}, got {}",
                    entry.entry_id,
                    crypto::to_hex(&prev_hash),
                    crypto::to_hex(&entry.previous_hash),
                ),
            });
        }

        // 2. Verify the entry's own hash.
        let computed = AuditEntry::compute_hash(&entry.data, &entry.previous_hash);
        if !crypto::constant_time_eq(&computed, &entry.entry_hash) {
            failures.push(VerificationFailure {
                entry_id: entry.entry_id,
                failure_type: FailureType::HashMismatch,
                message: format!(
                    "entry_hash mismatch at entry {}: expected {}, got {}",
                    entry.entry_id,
                    crypto::to_hex(&computed),
                    crypto::to_hex(&entry.entry_hash),
                ),
            });
        }

        prev_hash = entry.entry_hash;
    }

    let elapsed = start.elapsed().as_millis() as u64;
    let failure_count = failures.len() as u64;

    VerificationResult {
        valid: failures.is_empty(),
        entries_checked: entries.len() as u64,
        failures: failure_count,
        failure_details: failures,
        elapsed_ms: elapsed,
    }
}

/// Verify a single entry against its claimed `previous_hash`.
///
/// This does **not** verify the chain link (that `previous_hash` actually
/// matches the predecessor); use [`verify_chain`] for that.
pub fn verify_entry(entry: &AuditEntry) -> VerificationResult {
    let start = std::time::Instant::now();
    let mut failures = Vec::new();

    let computed = AuditEntry::compute_hash(&entry.data, &entry.previous_hash);
    if !crypto::constant_time_eq(&computed, &entry.entry_hash) {
        failures.push(VerificationFailure {
            entry_id: entry.entry_id,
            failure_type: FailureType::HashMismatch,
            message: format!(
                "entry_hash mismatch at entry {}: expected {}, got {}",
                entry.entry_id,
                crypto::to_hex(&computed),
                crypto::to_hex(&entry.entry_hash),
            ),
        });
    }

    let elapsed = start.elapsed().as_millis() as u64;
    VerificationResult {
        valid: failures.is_empty(),
        entries_checked: 1,
        failures: failures.len() as u64,
        failure_details: failures,
        elapsed_ms: elapsed,
    }
}

/// Verify the Merkle root for a batch of entries.
///
/// Recomputes the Merkle tree from the entry hashes and compares against the
/// stored root.
pub fn verify_batch(entries: &[AuditEntry], expected_root: &Hash256) -> VerificationResult {
    let start = std::time::Instant::now();
    let mut failures = Vec::new();

    let leaf_hashes: Vec<Hash256> = entries.iter().map(|e| e.entry_hash).collect();
    let computed_root = crypto::merkle_root(&leaf_hashes);

    if !crypto::constant_time_eq(&computed_root, expected_root) {
        failures.push(VerificationFailure {
            entry_id: entries.first().map(|e| e.entry_id).unwrap_or(0),
            failure_type: FailureType::MerkleRootMismatch,
            message: format!(
                "Merkle root mismatch: expected {}, computed {}",
                crypto::to_hex(expected_root),
                crypto::to_hex(&computed_root),
            ),
        });
    }

    // Also verify internal chain integrity.
    let chain_result = verify_chain(entries, None);
    for f in chain_result.failure_details {
        failures.push(f);
    }

    let elapsed = start.elapsed().as_millis() as u64;
    VerificationResult {
        valid: failures.is_empty(),
        entries_checked: entries.len() as u64,
        failures: failures.len() as u64,
        failure_details: failures,
        elapsed_ms: elapsed,
    }
}

/// Verify chain integrity over a time range.
///
/// This is a convenience wrapper: the caller provides entries that fall within
/// `[start, end)` and optionally the hash of the entry immediately preceding
/// the range.
pub fn verify_range(
    entries: &[AuditEntry],
    _start: DateTime<Utc>,
    _end: DateTime<Utc>,
    expected_previous_hash: Option<Hash256>,
) -> VerificationResult {
    verify_chain(entries, expected_previous_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::entry::{AuditEntryType, PendingEntry};
    use crate::ledger::merkle_chain::ChainBuilder;

    fn make_batch(n: usize) -> Vec<AuditEntry> {
        let mut builder = ChainBuilder::new();
        let pending: Vec<PendingEntry> = (0..n)
            .map(|i| PendingEntry {
                entry_type: AuditEntryType::ProbeResult,
                timestamp: Utc::now(),
                gpu_uuid: None,
                sm_id: None,
                data: format!("entry-{i}").into_bytes(),
            })
            .collect();
        builder.build_batch(pending).entries
    }

    #[test]
    fn test_verify_chain_valid() {
        let entries = make_batch(10);
        let result = verify_chain(&entries, Some(ZERO_HASH));
        assert!(result.valid);
        assert_eq!(result.entries_checked, 10);
    }

    #[test]
    fn test_verify_chain_detects_tamper() {
        let mut entries = make_batch(5);
        // Tamper with the data of entry 2.
        entries[2].data = b"tampered!".to_vec();
        let result = verify_chain(&entries, Some(ZERO_HASH));
        assert!(!result.valid);
        assert!(result.failures > 0);
    }

    #[test]
    fn test_verify_entry_valid() {
        let entries = make_batch(1);
        let result = verify_entry(&entries[0]);
        assert!(result.valid);
    }

    #[test]
    fn test_verify_batch_valid() {
        let mut builder = ChainBuilder::new();
        let pending: Vec<PendingEntry> = (0..4)
            .map(|i| PendingEntry {
                entry_type: AuditEntryType::ProbeResult,
                timestamp: Utc::now(),
                gpu_uuid: None,
                sm_id: None,
                data: format!("e{i}").into_bytes(),
            })
            .collect();
        let batch = builder.build_batch(pending);
        let result = verify_batch(&batch.entries, &batch.merkle_root);
        assert!(result.valid);
    }

    #[test]
    fn test_verify_empty() {
        let result = verify_chain(&[], None);
        assert!(result.valid);
        assert_eq!(result.entries_checked, 0);
    }
}
