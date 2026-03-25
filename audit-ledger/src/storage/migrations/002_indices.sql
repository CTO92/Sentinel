-- 002_indices.sql
-- Secondary indices for efficient querying of the audit ledger.

CREATE INDEX IF NOT EXISTS idx_audit_entries_gpu_uuid
    ON audit_entries (gpu_uuid)
    WHERE gpu_uuid IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_audit_entries_timestamp
    ON audit_entries (timestamp);

CREATE INDEX IF NOT EXISTS idx_audit_entries_entry_type
    ON audit_entries (entry_type);

CREATE INDEX IF NOT EXISTS idx_audit_entries_batch_sequence
    ON audit_entries (batch_sequence);

CREATE INDEX IF NOT EXISTS idx_audit_entries_gpu_timestamp
    ON audit_entries (gpu_uuid, timestamp)
    WHERE gpu_uuid IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_audit_entries_type_timestamp
    ON audit_entries (entry_type, timestamp);

CREATE INDEX IF NOT EXISTS idx_chain_anchors_batch_sequence
    ON chain_anchors (batch_sequence);

INSERT INTO schema_version (version) VALUES (2) ON CONFLICT DO NOTHING;
