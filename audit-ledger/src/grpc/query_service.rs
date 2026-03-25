//! gRPC service for querying the audit trail, verifying chain integrity,
//! generating compliance reports, and exporting data.

use std::sync::Arc;

use chrono::DateTime;
use tonic::{Request, Response, Status};
use tracing::error;

use crate::compliance::{export, iso27001_report, soc2_report};
use crate::grpc::proto::*;
use crate::grpc::server::SharedState;
use crate::ledger::entry::AuditEntryType;
use crate::ledger::verification;
use crate::storage::postgres::{self, AuditQuery};
use crate::util::metrics;

// ---------------------------------------------------------------------------
// Trait (would be tonic-generated)
// ---------------------------------------------------------------------------

#[tonic::async_trait]
pub trait QueryServiceRpc: Send + Sync + 'static {
    async fn query_audit_trail(
        &self,
        request: Request<AuditQueryProto>,
    ) -> Result<Response<AuditQueryResponse>, Status>;

    async fn verify_chain(
        &self,
        request: Request<VerifyChainRequest>,
    ) -> Result<Response<VerifyChainResponse>, Status>;

    async fn generate_report(
        &self,
        request: Request<GenerateReportRequest>,
    ) -> Result<Response<GenerateReportResponse>, Status>;

    async fn export_audit_trail(
        &self,
        request: Request<ExportRequest>,
    ) -> Result<Response<ExportResponse>, Status>;
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

pub struct AuditQueryService {
    state: Arc<SharedState>,
}

impl AuditQueryService {
    pub fn new(state: Arc<SharedState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl QueryServiceRpc for AuditQueryService {
    async fn query_audit_trail(
        &self,
        request: Request<AuditQueryProto>,
    ) -> Result<Response<AuditQueryResponse>, Status> {
        let _timer = metrics::QUERY_LATENCY_SECONDS.start_timer();
        let q = request.into_inner();

        let entry_type = q
            .entry_type
            .and_then(|v| AuditEntryType::from_i16(v as i16));

        let query = AuditQuery {
            gpu_uuid: q.gpu_uuid,
            entry_type,
            start_time: q
                .start_time_unix_ms
                .and_then(DateTime::from_timestamp_millis),
            end_time: q
                .end_time_unix_ms
                .and_then(DateTime::from_timestamp_millis),
            limit: if q.limit > 0 { Some(q.limit) } else { None },
            offset: if q.offset > 0 { Some(q.offset) } else { None },
        };

        let entries = postgres::query_entries(&self.state.pool, &query)
            .await
            .map_err(|e| {
                error!(error = %e, "query_audit_trail failed");
                Status::internal(format!("query failed: {e}"))
            })?;

        let proto_entries: Vec<AuditEntryProto> = entries
            .iter()
            .map(|e| AuditEntryProto {
                entry_type: e.entry_type.as_i16() as i32,
                timestamp_unix_ms: e.timestamp.timestamp_millis(),
                gpu_uuid: e.gpu_uuid.clone(),
                sm_id: e.sm_id,
                data: e.data.clone(),
            })
            .collect();

        let total = proto_entries.len() as u64;

        Ok(Response::new(AuditQueryResponse {
            entries: proto_entries,
            total_count: total,
        }))
    }

    async fn verify_chain(
        &self,
        request: Request<VerifyChainRequest>,
    ) -> Result<Response<VerifyChainResponse>, Status> {
        let req = request.into_inner();

        let query = AuditQuery {
            start_time: req
                .start_time_unix_ms
                .and_then(DateTime::from_timestamp_millis),
            end_time: req
                .end_time_unix_ms
                .and_then(DateTime::from_timestamp_millis),
            ..Default::default()
        };

        let entries = postgres::query_entries(&self.state.pool, &query)
            .await
            .map_err(|e| {
                error!(error = %e, "verify_chain query failed");
                Status::internal(format!("query failed: {e}"))
            })?;

        let result = verification::verify_chain(&entries, None);

        if result.valid {
            metrics::CHAIN_VERIFICATION_RESULT
                .with_label_values(&["valid"])
                .inc();
        } else {
            metrics::CHAIN_VERIFICATION_RESULT
                .with_label_values(&["invalid"])
                .inc();
        }

        let details_json =
            serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string());

        Ok(Response::new(VerifyChainResponse {
            valid: result.valid,
            entries_checked: result.entries_checked,
            failures: result.failures,
            details_json,
        }))
    }

    async fn generate_report(
        &self,
        request: Request<GenerateReportRequest>,
    ) -> Result<Response<GenerateReportResponse>, Status> {
        let req = request.into_inner();

        let start = DateTime::from_timestamp_millis(req.start_time_unix_ms)
            .ok_or_else(|| Status::invalid_argument("invalid start_time"))?;
        let end = DateTime::from_timestamp_millis(req.end_time_unix_ms)
            .ok_or_else(|| Status::invalid_argument("invalid end_time"))?;

        let json = match req.report_type.as_str() {
            "soc2" => {
                let report = soc2_report::generate(&self.state.pool, start, end)
                    .await
                    .map_err(|e| {
                        error!(error = %e, "SOC 2 report generation failed");
                        Status::internal(format!("report generation failed: {e}"))
                    })?;
                serde_json::to_string_pretty(&report)
                    .map_err(|e| Status::internal(e.to_string()))?
            }
            "iso27001" => {
                let report = iso27001_report::generate(&self.state.pool, start, end)
                    .await
                    .map_err(|e| {
                        error!(error = %e, "ISO 27001 report generation failed");
                        Status::internal(format!("report generation failed: {e}"))
                    })?;
                serde_json::to_string_pretty(&report)
                    .map_err(|e| Status::internal(e.to_string()))?
            }
            other => {
                return Err(Status::invalid_argument(format!(
                    "unknown report type: {other}"
                )));
            }
        };

        Ok(Response::new(GenerateReportResponse {
            report_json: json,
        }))
    }

    async fn export_audit_trail(
        &self,
        request: Request<ExportRequest>,
    ) -> Result<Response<ExportResponse>, Status> {
        let req = request.into_inner();

        let filter = export::ExportFilter {
            gpu_uuid: req.gpu_uuid,
            entry_type: req
                .entry_type
                .and_then(|v| AuditEntryType::from_i16(v as i16)),
            start_time: req
                .start_time_unix_ms
                .and_then(DateTime::from_timestamp_millis),
            end_time: req
                .end_time_unix_ms
                .and_then(DateTime::from_timestamp_millis),
        };

        let (payload, content_type) = match req.format.as_str() {
            "csv" => {
                let data = export::export_csv(&self.state.pool, filter)
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?;
                (data, "text/csv")
            }
            "json" => {
                let data = export::export_json(&self.state.pool, filter)
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?;
                (data, "application/json")
            }
            "html" => {
                let data = export::export_html(&self.state.pool, filter)
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?;
                (data, "text/html")
            }
            other => {
                return Err(Status::invalid_argument(format!(
                    "unknown export format: {other}"
                )));
            }
        };

        Ok(Response::new(ExportResponse {
            payload,
            content_type: content_type.into(),
        }))
    }
}

// ---------------------------------------------------------------------------
// tonic service wrapper (normally generated)
// ---------------------------------------------------------------------------
//
// `AuditQueryServer` would normally be generated by `tonic-build` from the
// `.proto` definition.  Until the proto compilation pipeline is in place, the
// service *implementation* above (`AuditQueryService`) is kept ready so that
// it can be plugged into the generated server type with no further changes.
