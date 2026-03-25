//! Audit entry types and structures for the SENTINEL ledger.
//!
//! Every event flowing through SENTINEL — probe results, anomaly detections,
//! quarantine actions, configuration changes — is represented as an [`AuditEntry`].
//! Entries are serialisable, hash-chainable, and stored immutably in PostgreSQL.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::util::crypto::{self, Hash256, ZERO_HASH};

// ---------------------------------------------------------------------------
// Entry type
// ---------------------------------------------------------------------------

/// Discriminator for the category of audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(i16)]
pub enum AuditEntryType {
    /// Result from a GPU integrity probe.
    ProbeResult = 0,
    /// Anomaly detected by the detection engine.
    AnomalyEvent = 1,
    /// GPU quarantine / un-quarantine action.
    QuarantineAction = 2,
    /// SENTINEL configuration change.
    ConfigChange = 3,
    /// Triple Modular Redundancy comparison result.
    TmrResult = 4,
    /// Generic system-level event (startup, shutdown, etc.).
    SystemEvent = 5,
}

impl AuditEntryType {
    /// Convert from the `SMALLINT` stored in PostgreSQL.
    pub fn from_i16(v: i16) -> Option<Self> {
        match v {
            0 => Some(Self::ProbeResult),
            1 => Some(Self::AnomalyEvent),
            2 => Some(Self::QuarantineAction),
            3 => Some(Self::ConfigChange),
            4 => Some(Self::TmrResult),
            5 => Some(Self::SystemEvent),
            _ => None,
        }
    }

    /// Return the i16 representation.
    pub fn as_i16(self) -> i16 {
        self as i16
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::ProbeResult => "ProbeResult",
            Self::AnomalyEvent => "AnomalyEvent",
            Self::QuarantineAction => "QuarantineAction",
            Self::ConfigChange => "ConfigChange",
            Self::TmrResult => "TmrResult",
            Self::SystemEvent => "SystemEvent",
        }
    }
}

// ---------------------------------------------------------------------------
// Audit entry
// ---------------------------------------------------------------------------

/// A single audit record in the SENTINEL ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Auto-incrementing identifier assigned by the database.
    pub entry_id: u64,
    /// Category of the event.
    pub entry_type: AuditEntryType,
    /// When the event occurred.
    pub timestamp: DateTime<Utc>,
    /// Optional GPU UUID (if the event is GPU-specific).
    pub gpu_uuid: Option<String>,
    /// Optional SM (Streaming Multiprocessor) identifier.
    pub sm_id: Option<i32>,
    /// Opaque serialised payload (JSON bytes, protobuf, etc.).
    pub data: Vec<u8>,
    /// SHA-256 hash of the preceding entry in the chain.
    pub previous_hash: Hash256,
    /// SHA-256 hash of **this** entry: `SHA256(data || previous_hash)`.
    pub entry_hash: Hash256,
    /// Merkle root — populated only at batch boundaries.
    pub merkle_root: Option<Hash256>,
    /// Batch sequence number this entry belongs to.
    pub batch_sequence: u64,
}

impl AuditEntry {
    /// Compute the hash for this entry given its data and the previous hash.
    ///
    /// ```text
    /// entry_hash = SHA-256( data || previous_hash )
    /// ```
    pub fn compute_hash(data: &[u8], previous_hash: &Hash256) -> Hash256 {
        crypto::chain_hash(data, previous_hash)
    }

    /// Verify that `self.entry_hash` matches the expected value.
    pub fn verify_hash(&self) -> bool {
        let expected = Self::compute_hash(&self.data, &self.previous_hash);
        crypto::constant_time_eq(&expected, &self.entry_hash)
    }
}

// ---------------------------------------------------------------------------
// Pending entry (pre-hash, pre-ID)
// ---------------------------------------------------------------------------

/// An event that has been received but not yet assigned an ID or chained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingEntry {
    pub entry_type: AuditEntryType,
    pub timestamp: DateTime<Utc>,
    pub gpu_uuid: Option<String>,
    pub sm_id: Option<i32>,
    pub data: Vec<u8>,
}

impl PendingEntry {
    /// Convert into a fully chained [`AuditEntry`].
    pub fn into_audit_entry(
        self,
        entry_id: u64,
        previous_hash: Hash256,
        batch_sequence: u64,
    ) -> AuditEntry {
        let entry_hash = AuditEntry::compute_hash(&self.data, &previous_hash);
        AuditEntry {
            entry_id,
            entry_type: self.entry_type,
            timestamp: self.timestamp,
            gpu_uuid: self.gpu_uuid,
            sm_id: self.sm_id,
            data: self.data,
            previous_hash,
            entry_hash,
            merkle_root: None,
            batch_sequence,
        }
    }
}

/// Create a genesis entry (the very first entry in a new ledger).
pub fn genesis_entry() -> AuditEntry {
    let data = b"SENTINEL Audit Ledger Genesis".to_vec();
    let entry_hash = AuditEntry::compute_hash(&data, &ZERO_HASH);
    AuditEntry {
        entry_id: 0,
        entry_type: AuditEntryType::SystemEvent,
        timestamp: Utc::now(),
        gpu_uuid: None,
        sm_id: None,
        data,
        previous_hash: ZERO_HASH,
        entry_hash,
        merkle_root: None,
        batch_sequence: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_type_roundtrip() {
        for v in 0..=5i16 {
            let t = AuditEntryType::from_i16(v).unwrap();
            assert_eq!(t.as_i16(), v);
        }
        assert!(AuditEntryType::from_i16(99).is_none());
    }

    #[test]
    fn test_compute_and_verify_hash() {
        let data = b"test payload".to_vec();
        let prev = ZERO_HASH;
        let hash = AuditEntry::compute_hash(&data, &prev);

        let entry = AuditEntry {
            entry_id: 1,
            entry_type: AuditEntryType::ProbeResult,
            timestamp: Utc::now(),
            gpu_uuid: None,
            sm_id: None,
            data,
            previous_hash: prev,
            entry_hash: hash,
            merkle_root: None,
            batch_sequence: 1,
        };
        assert!(entry.verify_hash());
    }

    #[test]
    fn test_pending_entry_conversion() {
        let pending = PendingEntry {
            entry_type: AuditEntryType::AnomalyEvent,
            timestamp: Utc::now(),
            gpu_uuid: Some("GPU-ABC".into()),
            sm_id: Some(4),
            data: b"anomaly data".to_vec(),
        };
        let entry = pending.into_audit_entry(42, ZERO_HASH, 1);
        assert_eq!(entry.entry_id, 42);
        assert!(entry.verify_hash());
    }

    #[test]
    fn test_genesis_entry_valid() {
        let g = genesis_entry();
        assert!(g.verify_hash());
        assert_eq!(g.previous_hash, ZERO_HASH);
    }
}
