//! Merkle hash-chain construction and batch processing.
//!
//! Entries form a sequential hash chain:
//!
//! ```text
//! Entry_N.hash = SHA-256( Entry_N.data || Entry_{N-1}.hash )
//! ```
//!
//! At batch boundaries a Merkle tree is computed over the entry hashes within
//! the batch, and the resulting root is stored alongside the entries.

use crate::ledger::entry::{AuditEntry, PendingEntry};
use crate::util::crypto::{self, Hash256, ZERO_HASH};

/// Default number of entries in a single batch.
pub const DEFAULT_BATCH_SIZE: usize = 1000;

/// A committed batch of audit entries with a Merkle root.
#[derive(Debug, Clone)]
pub struct CommittedBatch {
    /// The entries, fully chained and with the Merkle root set on the last entry.
    pub entries: Vec<AuditEntry>,
    /// The Merkle root over all entry hashes in this batch.
    pub merkle_root: Hash256,
    /// Monotonically increasing batch sequence number.
    pub batch_sequence: u64,
}

/// Builds batches from pending entries, maintaining chain state.
#[derive(Debug)]
pub struct ChainBuilder {
    /// Hash of the most recent entry in the chain (or ZERO_HASH for a new ledger).
    last_hash: Hash256,
    /// Next entry ID to assign (auto-increment surrogate).
    next_entry_id: u64,
    /// Next batch sequence number.
    next_batch_sequence: u64,
}

impl ChainBuilder {
    /// Create a new [`ChainBuilder`] starting from an empty chain.
    pub fn new() -> Self {
        Self {
            last_hash: ZERO_HASH,
            next_entry_id: 1,
            next_batch_sequence: 1,
        }
    }

    /// Resume a chain from existing state (e.g. after restart).
    pub fn resume(last_hash: Hash256, next_entry_id: u64, next_batch_sequence: u64) -> Self {
        Self {
            last_hash,
            next_entry_id,
            next_batch_sequence,
        }
    }

    /// Return the hash of the most recent entry.
    pub fn last_hash(&self) -> &Hash256 {
        &self.last_hash
    }

    /// Return the next batch sequence that will be assigned.
    pub fn next_batch_sequence(&self) -> u64 {
        self.next_batch_sequence
    }

    /// Build a committed batch from a list of pending entries.
    ///
    /// 1. Entries are sorted by timestamp.
    /// 2. Each entry is chained to its predecessor.
    /// 3. A Merkle tree is built over the entry hashes.
    /// 4. The Merkle root is attached to the final entry in the batch.
    pub fn build_batch(&mut self, mut pending: Vec<PendingEntry>) -> CommittedBatch {
        // Sort by timestamp for deterministic ordering.
        pending.sort_by_key(|e| e.timestamp);

        let batch_seq = self.next_batch_sequence;
        let mut entries = Vec::with_capacity(pending.len());
        let mut leaf_hashes = Vec::with_capacity(pending.len());

        for p in pending {
            let id = self.next_entry_id;
            self.next_entry_id += 1;

            let entry = p.into_audit_entry(id, self.last_hash, batch_seq);
            self.last_hash = entry.entry_hash;
            leaf_hashes.push(entry.entry_hash);
            entries.push(entry);
        }

        // Compute the Merkle root over this batch.
        let root = crypto::merkle_root(&leaf_hashes);

        // Tag the last entry with the root.
        if let Some(last) = entries.last_mut() {
            last.merkle_root = Some(root);
        }

        self.next_batch_sequence += 1;

        CommittedBatch {
            entries,
            merkle_root: root,
            batch_sequence: batch_seq,
        }
    }
}

impl Default for ChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Batch accumulator (time + count based flushing)
// ---------------------------------------------------------------------------

use std::time::{Duration, Instant};

/// Accumulates pending entries and signals when a batch should be flushed.
#[derive(Debug)]
pub struct BatchAccumulator {
    buffer: Vec<PendingEntry>,
    max_size: usize,
    max_age: Duration,
    created_at: Instant,
}

impl BatchAccumulator {
    /// Create a new accumulator with the given size and age limits.
    pub fn new(max_size: usize, max_age: Duration) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            max_size,
            max_age,
            created_at: Instant::now(),
        }
    }

    /// Create with default settings (1000 entries, 5 seconds).
    pub fn default_config() -> Self {
        Self::new(DEFAULT_BATCH_SIZE, Duration::from_secs(5))
    }

    /// Push an entry. Returns `true` if the batch should be flushed.
    pub fn push(&mut self, entry: PendingEntry) -> bool {
        self.buffer.push(entry);
        self.should_flush()
    }

    /// Check whether the batch should be flushed (size or age).
    pub fn should_flush(&self) -> bool {
        self.buffer.len() >= self.max_size || self.created_at.elapsed() >= self.max_age
    }

    /// Drain the buffer, resetting the accumulator.
    pub fn drain(&mut self) -> Vec<PendingEntry> {
        self.created_at = Instant::now();
        std::mem::take(&mut self.buffer)
    }

    /// Number of entries currently buffered.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::entry::AuditEntryType;
    use chrono::Utc;

    fn make_pending(data: &[u8]) -> PendingEntry {
        PendingEntry {
            entry_type: AuditEntryType::ProbeResult,
            timestamp: Utc::now(),
            gpu_uuid: None,
            sm_id: None,
            data: data.to_vec(),
        }
    }

    #[test]
    fn test_chain_builder_single_batch() {
        let mut builder = ChainBuilder::new();
        let entries: Vec<PendingEntry> = (0..5).map(|i| make_pending(&[i])).collect();
        let batch = builder.build_batch(entries);

        assert_eq!(batch.entries.len(), 5);
        assert_eq!(batch.batch_sequence, 1);

        // Verify chain integrity.
        let mut prev = ZERO_HASH;
        for e in &batch.entries {
            assert_eq!(e.previous_hash, prev);
            assert!(e.verify_hash());
            prev = e.entry_hash;
        }

        // Last entry has Merkle root.
        assert!(batch.entries.last().unwrap().merkle_root.is_some());
        assert_eq!(
            batch.entries.last().unwrap().merkle_root.unwrap(),
            batch.merkle_root
        );
    }

    #[test]
    fn test_chain_continuity_across_batches() {
        let mut builder = ChainBuilder::new();

        let batch1 = builder.build_batch(vec![make_pending(b"a"), make_pending(b"b")]);
        let batch2 = builder.build_batch(vec![make_pending(b"c")]);

        // First entry of batch2 should chain from the last entry of batch1.
        assert_eq!(
            batch2.entries[0].previous_hash,
            batch1.entries.last().unwrap().entry_hash,
        );
    }

    #[test]
    fn test_merkle_root_matches_leaf_hashes() {
        let mut builder = ChainBuilder::new();
        let entries: Vec<PendingEntry> = (0..4).map(|i| make_pending(&[i])).collect();
        let batch = builder.build_batch(entries);

        let leaf_hashes: Vec<Hash256> = batch.entries.iter().map(|e| e.entry_hash).collect();
        let expected_root = crypto::merkle_root(&leaf_hashes);
        assert_eq!(batch.merkle_root, expected_root);
    }

    #[test]
    fn test_batch_accumulator_size_trigger() {
        let mut acc = BatchAccumulator::new(3, Duration::from_secs(9999));
        assert!(!acc.push(make_pending(b"1")));
        assert!(!acc.push(make_pending(b"2")));
        assert!(acc.push(make_pending(b"3")));
        assert_eq!(acc.len(), 3);

        let drained = acc.drain();
        assert_eq!(drained.len(), 3);
        assert!(acc.is_empty());
    }
}
