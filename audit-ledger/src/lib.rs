//! SENTINEL Audit Ledger
//!
//! An immutable, verifiable record of all events flowing through the SENTINEL
//! Silent Data Corruption detection framework.  Events are hash-chained using
//! SHA-256, batched with Merkle roots, and stored in PostgreSQL with monthly
//! range partitioning.
//!
//! # Architecture
//!
//! ```text
//! Tokio runtime
//! ├── gRPC server (event ingestion)
//! ├── Batch accumulator (buffers events, flushes on count or time)
//! ├── Chain builder (computes hashes, writes to PostgreSQL)
//! ├── Verification worker (periodic chain integrity check)
//! ├── Query server (gRPC for audit trail queries)
//! └── Report generator (on-demand compliance reports)
//! ```

pub mod compliance;
pub mod grpc;
pub mod ledger;
pub mod storage;
pub mod util;
