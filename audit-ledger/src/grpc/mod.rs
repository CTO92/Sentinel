/// gRPC server setup and shared types.
pub mod server;

/// Event ingestion service.
pub mod ingest_service;

/// Query and reporting service.
pub mod query_service;

// ---------------------------------------------------------------------------
// Manually-defined protobuf message types (prost structs).
//
// In a production setting these would be generated from .proto files via
// tonic-build.  We define them inline so the crate compiles without an
// external proto repository.
// ---------------------------------------------------------------------------

/// Protobuf-compatible message types shared across gRPC services.
pub mod proto {
    use prost::Message;

    /// An audit entry as transmitted over gRPC.
    #[derive(Clone, PartialEq, Message)]
    pub struct AuditEntryProto {
        #[prost(int32, tag = "1")]
        pub entry_type: i32,
        #[prost(int64, tag = "2")]
        pub timestamp_unix_ms: i64,
        #[prost(string, optional, tag = "3")]
        pub gpu_uuid: Option<String>,
        #[prost(int32, optional, tag = "4")]
        pub sm_id: Option<i32>,
        #[prost(bytes = "vec", tag = "5")]
        pub data: Vec<u8>,
    }

    /// Acknowledgement for a batch of ingested events.
    #[derive(Clone, PartialEq, Message)]
    pub struct IngestAck {
        #[prost(uint64, tag = "1")]
        pub batch_sequence: u64,
        #[prost(uint64, tag = "2")]
        pub entry_count: u64,
        #[prost(bytes = "vec", tag = "3")]
        pub merkle_root: Vec<u8>,
    }

    /// Query parameters for the audit trail.
    #[derive(Clone, PartialEq, Message)]
    pub struct AuditQueryProto {
        #[prost(string, optional, tag = "1")]
        pub gpu_uuid: Option<String>,
        #[prost(int32, optional, tag = "2")]
        pub entry_type: Option<i32>,
        #[prost(int64, optional, tag = "3")]
        pub start_time_unix_ms: Option<i64>,
        #[prost(int64, optional, tag = "4")]
        pub end_time_unix_ms: Option<i64>,
        #[prost(int64, tag = "5")]
        pub limit: i64,
        #[prost(int64, tag = "6")]
        pub offset: i64,
    }

    /// A page of audit entries returned by a query.
    #[derive(Clone, PartialEq, Message)]
    pub struct AuditQueryResponse {
        #[prost(message, repeated, tag = "1")]
        pub entries: Vec<AuditEntryProto>,
        #[prost(uint64, tag = "2")]
        pub total_count: u64,
    }

    /// Request to verify chain integrity.
    #[derive(Clone, PartialEq, Message)]
    pub struct VerifyChainRequest {
        #[prost(int64, optional, tag = "1")]
        pub start_time_unix_ms: Option<i64>,
        #[prost(int64, optional, tag = "2")]
        pub end_time_unix_ms: Option<i64>,
    }

    /// Verification result.
    #[derive(Clone, PartialEq, Message)]
    pub struct VerifyChainResponse {
        #[prost(bool, tag = "1")]
        pub valid: bool,
        #[prost(uint64, tag = "2")]
        pub entries_checked: u64,
        #[prost(uint64, tag = "3")]
        pub failures: u64,
        #[prost(string, tag = "4")]
        pub details_json: String,
    }

    /// Request to generate a compliance report.
    #[derive(Clone, PartialEq, Message)]
    pub struct GenerateReportRequest {
        /// "soc2" or "iso27001".
        #[prost(string, tag = "1")]
        pub report_type: String,
        #[prost(int64, tag = "2")]
        pub start_time_unix_ms: i64,
        #[prost(int64, tag = "3")]
        pub end_time_unix_ms: i64,
    }

    /// Generated report.
    #[derive(Clone, PartialEq, Message)]
    pub struct GenerateReportResponse {
        #[prost(string, tag = "1")]
        pub report_json: String,
    }

    /// Export request.
    #[derive(Clone, PartialEq, Message)]
    pub struct ExportRequest {
        /// "csv", "json", or "html".
        #[prost(string, tag = "1")]
        pub format: String,
        #[prost(string, optional, tag = "2")]
        pub gpu_uuid: Option<String>,
        #[prost(int32, optional, tag = "3")]
        pub entry_type: Option<i32>,
        #[prost(int64, optional, tag = "4")]
        pub start_time_unix_ms: Option<i64>,
        #[prost(int64, optional, tag = "5")]
        pub end_time_unix_ms: Option<i64>,
    }

    /// Export response (raw bytes).
    #[derive(Clone, PartialEq, Message)]
    pub struct ExportResponse {
        #[prost(bytes = "vec", tag = "1")]
        pub payload: Vec<u8>,
        #[prost(string, tag = "2")]
        pub content_type: String,
    }
}
