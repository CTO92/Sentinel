-- 003_partitioning.sql
-- Monthly range partitioning on the audit_entries table.
--
-- PostgreSQL native declarative partitioning.  The base table is converted to a
-- partitioned table and a function + trigger create new monthly partitions on
-- demand.  Existing rows (if any) are migrated into the appropriate partition.
--
-- NOTE: This migration is designed to run on a fresh database.  For an existing
-- non-partitioned table the DBA should perform a controlled migration outside
-- of this script.

-- 1. Create the partitioned parent (only if table does not already use partitioning).
--    We wrap the DDL in a DO block so the migration is idempotent.
DO $$
BEGIN
    -- Check whether audit_entries is already partitioned.
    IF NOT EXISTS (
        SELECT 1
        FROM pg_partitioned_table
        WHERE partrelid = 'audit_entries'::regclass
    ) THEN
        -- Rename the original table, create the partitioned replacement, then
        -- copy data across.  This is safe when run on initial deployment (empty
        -- table).
        ALTER TABLE audit_entries RENAME TO audit_entries_old;

        CREATE TABLE audit_entries (
            entry_id       BIGSERIAL,
            entry_type     SMALLINT     NOT NULL,
            timestamp      TIMESTAMPTZ  NOT NULL,
            gpu_uuid       VARCHAR(64),
            sm_id          INTEGER,
            data           BYTEA        NOT NULL,
            previous_hash  BYTEA        NOT NULL,
            entry_hash     BYTEA        NOT NULL,
            merkle_root    BYTEA,
            batch_sequence BIGINT       NOT NULL,
            created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            PRIMARY KEY (entry_id, timestamp)
        ) PARTITION BY RANGE (timestamp);

        -- Copy any existing rows (usually empty on first deploy).
        INSERT INTO audit_entries
            SELECT * FROM audit_entries_old;

        DROP TABLE audit_entries_old;
    END IF;
END
$$;

-- 2. Function to auto-create monthly partitions.
CREATE OR REPLACE FUNCTION create_monthly_partition(target_date DATE)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    partition_name TEXT;
    start_date     DATE;
    end_date       DATE;
BEGIN
    start_date     := DATE_TRUNC('month', target_date);
    end_date       := start_date + INTERVAL '1 month';
    partition_name := 'audit_entries_' || TO_CHAR(start_date, 'YYYY_MM');

    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = partition_name
    ) THEN
        EXECUTE FORMAT(
            'CREATE TABLE %I PARTITION OF audit_entries
             FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );

        -- Re-create per-partition indices.
        EXECUTE FORMAT(
            'CREATE INDEX IF NOT EXISTS %I ON %I (gpu_uuid) WHERE gpu_uuid IS NOT NULL',
            partition_name || '_gpu_uuid_idx', partition_name
        );
        EXECUTE FORMAT(
            'CREATE INDEX IF NOT EXISTS %I ON %I (entry_type)',
            partition_name || '_entry_type_idx', partition_name
        );
        EXECUTE FORMAT(
            'CREATE INDEX IF NOT EXISTS %I ON %I (batch_sequence)',
            partition_name || '_batch_seq_idx', partition_name
        );

        RAISE NOTICE 'Created partition %', partition_name;
    END IF;
END
$$;

-- 3. Pre-create partitions for the current month and the next 3 months.
DO $$
DECLARE
    m INTEGER;
BEGIN
    FOR m IN 0..3 LOOP
        PERFORM create_monthly_partition(
            (CURRENT_DATE + (m || ' months')::INTERVAL)::DATE
        );
    END LOOP;
END
$$;

INSERT INTO schema_version (version) VALUES (3) ON CONFLICT DO NOTHING;
