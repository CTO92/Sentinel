//! ISO 27001 compliance report generator.
//!
//! Maps SENTINEL audit events to relevant ISO 27001:2022 Annex A controls:
//!
//! - **A.8 (Asset Management)**: GPU lifecycle tracking — registration, health
//!   status, quarantine and restoration.
//! - **A.12 (Operations Security)**: Anomaly detection, incident response, and
//!   monitoring effectiveness.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::PgPool;

use crate::ledger::entry::AuditEntryType;
use crate::storage::postgres::{self, AuditQuery};

/// A complete ISO 27001 compliance report.
#[derive(Debug, Clone, Serialize)]
pub struct Iso27001Report {
    pub metadata: ReportMetadata,
    pub a8_asset_management: AssetManagementControl,
    pub a12_operations_security: OperationsSecurityControl,
    pub summary: ReportSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportMetadata {
    pub report_type: String,
    pub standard: String,
    pub generated_at: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub generated_by: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AssetManagementControl {
    pub control_id: String,
    pub control_title: String,
    pub description: String,
    /// Unique GPU UUIDs observed in the period.
    pub gpus_observed: Vec<String>,
    /// Total events related to specific GPUs.
    pub gpu_events_total: u64,
    /// Quarantine lifecycle events.
    pub quarantine_actions: u64,
    /// Configuration changes affecting assets.
    pub config_changes: u64,
    pub findings: Vec<Finding>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OperationsSecurityControl {
    pub control_id: String,
    pub control_title: String,
    pub description: String,
    /// Total anomalies detected.
    pub anomalies_detected: u64,
    /// Probe results processed.
    pub probes_executed: u64,
    /// TMR comparisons performed.
    pub tmr_comparisons: u64,
    /// Mean time to detect (seconds) — from probe to anomaly event.
    pub mean_time_to_detect_seconds: Option<f64>,
    /// Incidents (anomalies that led to quarantine).
    pub incidents_count: u64,
    pub findings: Vec<Finding>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub severity: Severity,
    pub description: String,
    pub evidence_entry_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportSummary {
    pub total_events: u64,
    pub chain_integrity_verified: bool,
    pub controls_assessed: u32,
    pub findings_total: u64,
}

/// Generate an ISO 27001 report for the given period.
pub async fn generate(
    pool: &PgPool,
    period_start: DateTime<Utc>,
    period_end: DateTime<Utc>,
) -> Result<Iso27001Report> {
    // ----- A.8 Asset Management -----

    // Collect all GPU-related events.
    let all_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await
    .context("querying all entries")?;

    let mut gpu_set = std::collections::HashSet::new();
    let mut gpu_event_count = 0u64;
    for e in &all_entries {
        if let Some(ref uuid) = e.gpu_uuid {
            gpu_set.insert(uuid.clone());
            gpu_event_count += 1;
        }
    }
    let gpus_observed: Vec<String> = {
        let mut v: Vec<String> = gpu_set.into_iter().collect();
        v.sort();
        v
    };

    let quarantine_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::QuarantineAction),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await?;

    let config_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::ConfigChange),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await?;

    let mut a8_findings = Vec::new();
    if gpus_observed.is_empty() {
        a8_findings.push(Finding {
            severity: Severity::Info,
            description: "No GPU assets observed in the reporting period.".into(),
            evidence_entry_ids: vec![],
        });
    }

    let a8 = AssetManagementControl {
        control_id: "A.8".into(),
        control_title: "Asset Management".into(),
        description: "Identification, classification, and lifecycle management of GPU assets \
                       within the SENTINEL-monitored cluster."
            .into(),
        gpus_observed,
        gpu_events_total: gpu_event_count,
        quarantine_actions: quarantine_entries.len() as u64,
        config_changes: config_entries.len() as u64,
        findings: a8_findings,
    };

    // ----- A.12 Operations Security -----

    let anomaly_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::AnomalyEvent),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await?;

    let probe_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::ProbeResult),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await?;

    let tmr_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::TmrResult),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await?;

    // Estimate incidents: anomalies with a matching GPU that also has a quarantine.
    let quarantined_gpus: std::collections::HashSet<String> = quarantine_entries
        .iter()
        .filter_map(|e| e.gpu_uuid.clone())
        .collect();
    let incidents_count = anomaly_entries
        .iter()
        .filter(|e| e.gpu_uuid.as_ref().map_or(false, |g| quarantined_gpus.contains(g)))
        .count() as u64;

    let mut a12_findings = Vec::new();
    let anomaly_rate = if probe_entries.is_empty() {
        0.0
    } else {
        anomaly_entries.len() as f64 / probe_entries.len() as f64
    };
    if anomaly_rate > 0.01 {
        a12_findings.push(Finding {
            severity: Severity::Medium,
            description: format!(
                "Anomaly rate ({:.4}) exceeds 1% threshold. Investigation recommended.",
                anomaly_rate
            ),
            evidence_entry_ids: anomaly_entries.iter().take(10).map(|e| e.entry_id).collect(),
        });
    }

    let a12 = OperationsSecurityControl {
        control_id: "A.12".into(),
        control_title: "Operations Security".into(),
        description:
            "Monitoring, anomaly detection, and incident response for GPU compute operations."
                .into(),
        anomalies_detected: anomaly_entries.len() as u64,
        probes_executed: probe_entries.len() as u64,
        tmr_comparisons: tmr_entries.len() as u64,
        mean_time_to_detect_seconds: None, // Requires cross-referencing probe/anomaly timestamps per GPU.
        incidents_count,
        findings: a12_findings,
    };

    // ----- Integrity check -----
    let chain_result = crate::ledger::verification::verify_chain(&all_entries, None);

    let total_findings = (a8.findings.len() + a12.findings.len()) as u64;

    let summary = ReportSummary {
        total_events: all_entries.len() as u64,
        chain_integrity_verified: chain_result.valid,
        controls_assessed: 2,
        findings_total: total_findings,
    };

    let metadata = ReportMetadata {
        report_type: "ISO 27001 Internal Audit".into(),
        standard: "ISO/IEC 27001:2022".into(),
        generated_at: Utc::now(),
        period_start,
        period_end,
        generated_by: "SENTINEL Audit Ledger".into(),
    };

    crate::util::metrics::COMPLIANCE_REPORTS_TOTAL.inc();

    Ok(Iso27001Report {
        metadata,
        a8_asset_management: a8,
        a12_operations_security: a12,
        summary,
    })
}
