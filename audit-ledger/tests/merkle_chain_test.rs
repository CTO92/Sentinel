//! Integration-style tests for the Merkle hash chain.

use chrono::Utc;
use sentinel_audit_ledger::ledger::entry::{AuditEntryType, PendingEntry};
use sentinel_audit_ledger::ledger::merkle_chain::{BatchAccumulator, ChainBuilder, DEFAULT_BATCH_SIZE};
use sentinel_audit_ledger::ledger::verification;
use sentinel_audit_ledger::util::crypto::{self, Hash256, ZERO_HASH};
use std::time::Duration;

fn pending(data: &[u8]) -> PendingEntry {
    PendingEntry {
        entry_type: AuditEntryType::ProbeResult,
        timestamp: Utc::now(),
        gpu_uuid: Some("GPU-TEST-001".into()),
        sm_id: Some(0),
        data: data.to_vec(),
    }
}

#[test]
fn chain_of_one_entry() {
    let mut builder = ChainBuilder::new();
    let batch = builder.build_batch(vec![pending(b"single")]);
    assert_eq!(batch.entries.len(), 1);

    let e = &batch.entries[0];
    assert_eq!(e.previous_hash, ZERO_HASH);
    assert!(e.verify_hash());
    assert_eq!(e.merkle_root, Some(e.entry_hash));
}

#[test]
fn chain_multiple_batches_continuous() {
    let mut builder = ChainBuilder::new();

    let b1 = builder.build_batch(vec![pending(b"a"), pending(b"b"), pending(b"c")]);
    let b2 = builder.build_batch(vec![pending(b"d"), pending(b"e")]);
    let b3 = builder.build_batch(vec![pending(b"f")]);

    // Verify batch 1 internally.
    let res1 = verification::verify_chain(&b1.entries, Some(ZERO_HASH));
    assert!(res1.valid, "batch 1 chain invalid: {:?}", res1.failure_details);

    // Verify batch 2 chains from batch 1.
    let last_b1_hash = b1.entries.last().unwrap().entry_hash;
    let res2 = verification::verify_chain(&b2.entries, Some(last_b1_hash));
    assert!(res2.valid, "batch 2 chain invalid: {:?}", res2.failure_details);

    // Verify the whole sequence end-to-end.
    let mut all: Vec<_> = b1.entries.clone();
    all.extend(b2.entries.clone());
    all.extend(b3.entries.clone());
    let res_all = verification::verify_chain(&all, Some(ZERO_HASH));
    assert!(res_all.valid, "full chain invalid: {:?}", res_all.failure_details);
    assert_eq!(res_all.entries_checked, 6);
}

#[test]
fn merkle_root_correctness() {
    let mut builder = ChainBuilder::new();
    let entries: Vec<PendingEntry> = (0..8u8).map(|i| pending(&[i])).collect();
    let batch = builder.build_batch(entries);

    // Recompute Merkle root manually.
    let leaf_hashes: Vec<Hash256> = batch.entries.iter().map(|e| e.entry_hash).collect();
    let expected = crypto::merkle_root(&leaf_hashes);
    assert_eq!(batch.merkle_root, expected);

    // Verify via the verification module.
    let res = verification::verify_batch(&batch.entries, &batch.merkle_root);
    assert!(res.valid);
}

#[test]
fn tampered_entry_detected() {
    let mut builder = ChainBuilder::new();
    let batch = builder.build_batch(vec![pending(b"x"), pending(b"y"), pending(b"z")]);
    let mut entries = batch.entries;

    // Tamper with entry 1's data.
    entries[1].data = b"TAMPERED".to_vec();

    let res = verification::verify_chain(&entries, Some(ZERO_HASH));
    assert!(!res.valid);
    // Should report at least a hash mismatch on entry 1.
    assert!(res.failure_details.iter().any(|f| f.entry_id == entries[1].entry_id));
}

#[test]
fn chain_break_detected() {
    let mut builder = ChainBuilder::new();
    let batch = builder.build_batch(vec![pending(b"a"), pending(b"b"), pending(b"c")]);
    let mut entries = batch.entries;

    // Break the chain: change previous_hash of entry 2 to zero.
    entries[2].previous_hash = ZERO_HASH;
    // Recompute entry_hash so the self-hash check passes but chain link fails.
    entries[2].entry_hash =
        sentinel_audit_ledger::ledger::entry::AuditEntry::compute_hash(
            &entries[2].data,
            &entries[2].previous_hash,
        );

    let res = verification::verify_chain(&entries, Some(ZERO_HASH));
    assert!(!res.valid);
}

#[test]
fn merkle_root_mismatch_detected() {
    let mut builder = ChainBuilder::new();
    let batch = builder.build_batch(vec![pending(b"m"), pending(b"n")]);

    let wrong_root = crypto::sha256(b"wrong");
    let res = verification::verify_batch(&batch.entries, &wrong_root);
    assert!(!res.valid);
    assert!(res
        .failure_details
        .iter()
        .any(|f| matches!(f.failure_type, verification::FailureType::MerkleRootMismatch)));
}

#[test]
fn batch_accumulator_time_based_flush() {
    let mut acc = BatchAccumulator::new(9999, Duration::from_millis(1));
    acc.push(pending(b"1"));
    // Wait for the age to expire.
    std::thread::sleep(Duration::from_millis(5));
    assert!(acc.should_flush());
    let drained = acc.drain();
    assert_eq!(drained.len(), 1);
}

#[test]
fn large_batch_stress() {
    let mut builder = ChainBuilder::new();
    let entries: Vec<PendingEntry> = (0..DEFAULT_BATCH_SIZE)
        .map(|i| {
            PendingEntry {
                entry_type: AuditEntryType::ProbeResult,
                timestamp: Utc::now(),
                gpu_uuid: Some(format!("GPU-{:04}", i % 16)),
                sm_id: Some((i % 128) as i32),
                data: format!("probe-result-{i}").into_bytes(),
            }
        })
        .collect();

    let batch = builder.build_batch(entries);
    assert_eq!(batch.entries.len(), DEFAULT_BATCH_SIZE);

    let res = verification::verify_chain(&batch.entries, Some(ZERO_HASH));
    assert!(res.valid);
    assert_eq!(res.entries_checked, DEFAULT_BATCH_SIZE as u64);

    let res_merkle = verification::verify_batch(&batch.entries, &batch.merkle_root);
    assert!(res_merkle.valid);
}
