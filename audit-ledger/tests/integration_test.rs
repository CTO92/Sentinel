//! End-to-end integration tests for the audit ledger.
//!
//! These tests exercise the full pipeline without a database: chain building,
//! verification, Merkle roots, and export formatting.

use chrono::Utc;
use sentinel_audit_ledger::ledger::entry::{
    genesis_entry, AuditEntry, AuditEntryType, PendingEntry,
};
use sentinel_audit_ledger::ledger::merkle_chain::ChainBuilder;
use sentinel_audit_ledger::ledger::verification;
use sentinel_audit_ledger::util::crypto::{self, Hash256, ZERO_HASH};

/// Simulate a realistic event stream: probe results, anomaly, quarantine.
#[test]
fn realistic_event_stream() {
    let mut builder = ChainBuilder::new();

    // Batch 1: probe results from multiple GPUs.
    let probes: Vec<PendingEntry> = (0..10)
        .map(|i| PendingEntry {
            entry_type: AuditEntryType::ProbeResult,
            timestamp: Utc::now(),
            gpu_uuid: Some(format!("GPU-{:04}", i % 4)),
            sm_id: Some(i as i32),
            data: serde_json::to_vec(&serde_json::json!({
                "gpu": format!("GPU-{:04}", i % 4),
                "sm_id": i,
                "pattern": "walking_ones",
                "result": "pass",
            }))
            .unwrap(),
        })
        .collect();
    let batch1 = builder.build_batch(probes);

    // Batch 2: an anomaly and a quarantine action.
    let events = vec![
        PendingEntry {
            entry_type: AuditEntryType::AnomalyEvent,
            timestamp: Utc::now(),
            gpu_uuid: Some("GPU-0002".into()),
            sm_id: Some(7),
            data: serde_json::to_vec(&serde_json::json!({
                "gpu": "GPU-0002",
                "anomaly_type": "bit_flip",
                "confidence": 0.97,
            }))
            .unwrap(),
        },
        PendingEntry {
            entry_type: AuditEntryType::QuarantineAction,
            timestamp: Utc::now(),
            gpu_uuid: Some("GPU-0002".into()),
            sm_id: None,
            data: serde_json::to_vec(&serde_json::json!({
                "gpu": "GPU-0002",
                "action": "quarantine",
                "reason": "SDC anomaly detected",
            }))
            .unwrap(),
        },
    ];
    let batch2 = builder.build_batch(events);

    // Batch 3: config change and TMR result.
    let misc = vec![
        PendingEntry {
            entry_type: AuditEntryType::ConfigChange,
            timestamp: Utc::now(),
            gpu_uuid: None,
            sm_id: None,
            data: serde_json::to_vec(&serde_json::json!({
                "field": "probe_interval_ms",
                "old_value": 60000,
                "new_value": 30000,
            }))
            .unwrap(),
        },
        PendingEntry {
            entry_type: AuditEntryType::TmrResult,
            timestamp: Utc::now(),
            gpu_uuid: Some("GPU-0001".into()),
            sm_id: Some(0),
            data: serde_json::to_vec(&serde_json::json!({
                "gpu": "GPU-0001",
                "voter_result": "unanimous",
                "result_hash": "abcdef1234",
            }))
            .unwrap(),
        },
    ];
    let batch3 = builder.build_batch(misc);

    // Verify the entire chain end-to-end.
    let mut all_entries = Vec::new();
    all_entries.extend(batch1.entries.clone());
    all_entries.extend(batch2.entries.clone());
    all_entries.extend(batch3.entries.clone());

    let result = verification::verify_chain(&all_entries, Some(ZERO_HASH));
    assert!(result.valid, "full chain verification failed: {:?}", result.failure_details);
    assert_eq!(result.entries_checked, 14);

    // Verify each batch's Merkle root independently.
    let r1 = verification::verify_batch(&batch1.entries, &batch1.merkle_root);
    assert!(r1.valid);
    let r2 = verification::verify_batch(&batch2.entries, &batch2.merkle_root);
    assert!(r2.valid);
    let r3 = verification::verify_batch(&batch3.entries, &batch3.merkle_root);
    assert!(r3.valid);
}

/// Verify that the genesis entry is valid.
#[test]
fn genesis_entry_integrity() {
    let g = genesis_entry();
    assert!(g.verify_hash());
    assert_eq!(g.previous_hash, ZERO_HASH);
    assert_eq!(g.entry_type, AuditEntryType::SystemEvent);
}

/// Chain resume: simulate a restart by creating a new ChainBuilder from saved state.
#[test]
fn chain_resume_continuity() {
    let mut builder = ChainBuilder::new();
    let batch1 = builder.build_batch(vec![PendingEntry {
        entry_type: AuditEntryType::ProbeResult,
        timestamp: Utc::now(),
        gpu_uuid: None,
        sm_id: None,
        data: b"first".to_vec(),
    }]);

    // Simulate saving state and resuming.
    let last_hash = batch1.entries.last().unwrap().entry_hash;
    let next_id = batch1.entries.last().unwrap().entry_id + 1;
    let next_batch = builder.next_batch_sequence();

    let mut resumed = ChainBuilder::resume(last_hash, next_id, next_batch);
    let batch2 = resumed.build_batch(vec![PendingEntry {
        entry_type: AuditEntryType::SystemEvent,
        timestamp: Utc::now(),
        gpu_uuid: None,
        sm_id: None,
        data: b"second".to_vec(),
    }]);

    // The chain should be continuous.
    assert_eq!(batch2.entries[0].previous_hash, last_hash);
    assert!(batch2.entries[0].verify_hash());

    let mut all = batch1.entries.clone();
    all.extend(batch2.entries);
    let res = verification::verify_chain(&all, Some(ZERO_HASH));
    assert!(res.valid);
}

/// Hash chain is order-dependent: reordering entries invalidates the chain.
#[test]
fn reordering_breaks_chain() {
    let mut builder = ChainBuilder::new();
    let batch = builder.build_batch(vec![
        PendingEntry {
            entry_type: AuditEntryType::ProbeResult,
            timestamp: Utc::now(),
            gpu_uuid: None,
            sm_id: None,
            data: b"alpha".to_vec(),
        },
        PendingEntry {
            entry_type: AuditEntryType::ProbeResult,
            timestamp: Utc::now(),
            gpu_uuid: None,
            sm_id: None,
            data: b"beta".to_vec(),
        },
    ]);

    let mut entries = batch.entries;
    entries.swap(0, 1);

    let res = verification::verify_chain(&entries, Some(ZERO_HASH));
    assert!(!res.valid);
}

/// Verify that single-entry verification works.
#[test]
fn single_entry_verification() {
    let mut builder = ChainBuilder::new();
    let batch = builder.build_batch(vec![PendingEntry {
        entry_type: AuditEntryType::ProbeResult,
        timestamp: Utc::now(),
        gpu_uuid: None,
        sm_id: None,
        data: b"test".to_vec(),
    }]);

    let res = verification::verify_entry(&batch.entries[0]);
    assert!(res.valid);
    assert_eq!(res.entries_checked, 1);
}

/// Export row conversion preserves data.
#[test]
fn export_row_preserves_data() {
    use sentinel_audit_ledger::compliance::export::ExportRow;

    let data = b"hello world".to_vec();
    let hash = AuditEntry::compute_hash(&data, &ZERO_HASH);
    let entry = AuditEntry {
        entry_id: 99,
        entry_type: AuditEntryType::AnomalyEvent,
        timestamp: Utc::now(),
        gpu_uuid: Some("GPU-TEST".into()),
        sm_id: Some(42),
        data: data.clone(),
        previous_hash: ZERO_HASH,
        entry_hash: hash,
        merkle_root: None,
        batch_sequence: 5,
    };

    let row = ExportRow::from(&entry);
    assert_eq!(row.entry_id, 99);
    assert_eq!(row.batch_sequence, 5);

    // Verify data round-trips through base64.
    use base64::Engine;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&row.data_base64)
        .unwrap();
    assert_eq!(decoded, data);

    // Verify hash round-trips through hex.
    let decoded_hash = hex::decode(&row.entry_hash_hex).unwrap();
    assert_eq!(decoded_hash, hash.to_vec());
}
