-- 001_initial.sql
-- Creates the core tables for the SENTINEL Audit Ledger.

CREATE TABLE IF NOT EXISTS audit_entries (
    entry_id    BIGSERIAL    PRIMARY KEY,
    entry_type  SMALLINT     NOT NULL,
    timestamp   TIMESTAMPTZ  NOT NULL,
    gpu_uuid    VARCHAR(64),
    sm_id       INTEGER,
    data        BYTEA        NOT NULL,
    previous_hash BYTEA      NOT NULL,
    entry_hash  BYTEA        NOT NULL,
    merkle_root BYTEA,
    batch_sequence BIGINT    NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS merkle_roots (
    batch_sequence BIGSERIAL PRIMARY KEY,
    merkle_root    BYTEA     NOT NULL,
    entry_count    INTEGER   NOT NULL,
    first_entry_id BIGINT    NOT NULL,
    last_entry_id  BIGINT    NOT NULL,
    timestamp      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chain_anchors (
    anchor_id         BIGSERIAL    PRIMARY KEY,
    batch_sequence    BIGINT       NOT NULL,
    merkle_root       BYTEA        NOT NULL,
    external_timestamp VARCHAR(256),
    anchor_service    VARCHAR(128),
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Schema‐version tracking (idempotent).
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER      PRIMARY KEY,
    applied_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

INSERT INTO schema_version (version) VALUES (1) ON CONFLICT DO NOTHING;
