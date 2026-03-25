//! SOC 2 compliance report generator.
//!
//! Maps SENTINEL audit events to relevant SOC 2 Trust Service Criteria:
//!
//! - **A1.2 (Availability)**: system uptime, quarantine events.
//! - **CC7.2 (Security)**: security event monitoring, anomaly detection.
//! - **CC8.1 (Change Management)**: configuration changes.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::PgPool;

use crate::ledger::entry::AuditEntryType;
use crate::storage::postgres::{self, AuditQuery};

/// A complete SOC 2 compliance report.
#[derive(Debug, Clone, Serialize)]
pub struct Soc2Report {
    /// Report generation metadata.
    pub metadata: ReportMetadata,
    /// Control A1.2: System Availability.
    pub a1_2_availability: AvailabilityControl,
    /// Control CC7.2: Security Event Monitoring.
    pub cc7_2_security_monitoring: SecurityMonitoringControl,
    /// Control CC8.1: Change Management.
    pub cc8_1_change_management: ChangeManagementControl,
    /// Summary statistics.
    pub summary: ReportSummary,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportMetadata {
    pub report_type: String,
    pub generated_at: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub generated_by: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AvailabilityControl {
    pub control_id: String,
    pub control_description: String,
    pub total_system_events: u64,
    pub quarantine_events: u64,
    pub unquarantine_events: u64,
    pub gpu_availability_ratio: f64,
    pub events: Vec<EventSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SecurityMonitoringControl {
    pub control_id: String,
    pub control_description: String,
    pub total_anomalies_detected: u64,
    pub total_probe_results: u64,
    pub anomaly_rate: f64,
    pub tmr_comparisons: u64,
    pub events: Vec<EventSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChangeManagementControl {
    pub control_id: String,
    pub control_description: String,
    pub total_config_changes: u64,
    pub events: Vec<EventSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventSummary {
    pub entry_id: u64,
    pub entry_type: String,
    pub timestamp: DateTime<Utc>,
    pub gpu_uuid: Option<String>,
    pub data_preview: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReportSummary {
    pub total_events_in_period: u64,
    pub integrity_verified: bool,
    pub chain_continuous: bool,
}

/// Generate a SOC 2 compliance report for the given time period.
pub async fn generate(
    pool: &PgPool,
    period_start: DateTime<Utc>,
    period_end: DateTime<Utc>,
) -> Result<Soc2Report> {
    // A1.2 — Availability: quarantine actions and system events.
    let quarantine_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::QuarantineAction),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await
    .context("querying quarantine events")?;

    let system_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::SystemEvent),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await
    .context("querying system events")?;

    let quarantine_count = quarantine_entries.len() as u64;
    // Estimate un-quarantines as entries whose data contains "unquarantine".
    let unquarantine_count = quarantine_entries
        .iter()
        .filter(|e| {
            String::from_utf8_lossy(&e.data)
                .to_lowercase()
                .contains("unquarantine")
        })
        .count() as u64;
    let pure_quarantine = quarantine_count.saturating_sub(unquarantine_count);

    let availability_events: Vec<EventSummary> = quarantine_entries
        .iter()
        .chain(system_entries.iter())
        .map(to_event_summary)
        .collect();

    let gpu_availability_ratio = if quarantine_count == 0 {
        1.0
    } else {
        // Simplified: ratio of un-quarantines to total quarantine actions.
        unquarantine_count as f64 / quarantine_count.max(1) as f64
    };

    let a1_2 = AvailabilityControl {
        control_id: "A1.2".into(),
        control_description:
            "The entity authorizes, designs, develops or acquires, implements, operates, \
             approves, maintains, and monitors environmental protections, software, \
             data backup processes, and recovery infrastructure to meet its objectives."
                .into(),
        total_system_events: system_entries.len() as u64,
        quarantine_events: pure_quarantine,
        unquarantine_events: unquarantine_count,
        gpu_availability_ratio,
        events: availability_events,
    };

    // CC7.2 — Security monitoring.
    let anomaly_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::AnomalyEvent),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await
    .context("querying anomaly events")?;

    let probe_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::ProbeResult),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await
    .context("querying probe results")?;

    let tmr_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::TmrResult),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await
    .context("querying TMR results")?;

    let total_anomalies = anomaly_entries.len() as u64;
    let total_probes = probe_entries.len() as u64;
    let anomaly_rate = if total_probes == 0 {
        0.0
    } else {
        total_anomalies as f64 / total_probes as f64
    };

    let security_events: Vec<EventSummary> = anomaly_entries
        .iter()
        .map(to_event_summary)
        .collect();

    let cc7_2 = SecurityMonitoringControl {
        control_id: "CC7.2".into(),
        control_description:
            "The entity monitors system components and the operation of those components \
             for anomalies that are indicative of malicious acts, natural disasters, \
             and errors affecting the entity's ability to meet its objectives."
                .into(),
        total_anomalies_detected: total_anomalies,
        total_probe_results: total_probes,
        anomaly_rate,
        tmr_comparisons: tmr_entries.len() as u64,
        events: security_events,
    };

    // CC8.1 — Change management.
    let config_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            entry_type: Some(AuditEntryType::ConfigChange),
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await
    .context("querying config changes")?;

    let change_events: Vec<EventSummary> = config_entries
        .iter()
        .map(to_event_summary)
        .collect();

    let cc8_1 = ChangeManagementControl {
        control_id: "CC8.1".into(),
        control_description:
            "The entity authorizes, designs, develops or acquires, configures, documents, \
             tests, approves, and implements changes to infrastructure, data, software, \
             and procedures to meet its objectives."
                .into(),
        total_config_changes: config_entries.len() as u64,
        events: change_events,
    };

    // Total events in period.
    let all_entries = postgres::query_entries(
        pool,
        &AuditQuery {
            start_time: Some(period_start),
            end_time: Some(period_end),
            ..Default::default()
        },
    )
    .await
    .context("querying all events")?;

    // Quick chain verification on the returned entries.
    let chain_result = crate::ledger::verification::verify_chain(&all_entries, None);

    let summary = ReportSummary {
        total_events_in_period: all_entries.len() as u64,
        integrity_verified: chain_result.valid,
        chain_continuous: chain_result.valid,
    };

    let metadata = ReportMetadata {
        report_type: "SOC 2 Type II".into(),
        generated_at: Utc::now(),
        period_start,
        period_end,
        generated_by: "SENTINEL Audit Ledger".into(),
    };

    crate::util::metrics::COMPLIANCE_REPORTS_TOTAL.inc();

    Ok(Soc2Report {
        metadata,
        a1_2_availability: a1_2,
        cc7_2_security_monitoring: cc7_2,
        cc8_1_change_management: cc8_1,
        summary,
    })
}

fn to_event_summary(entry: &crate::ledger::entry::AuditEntry) -> EventSummary {
    let preview = String::from_utf8_lossy(&entry.data);
    let truncated = if preview.len() > 200 {
        format!("{}...", &preview[..200])
    } else {
        preview.to_string()
    };
    EventSummary {
        entry_id: entry.entry_id,
        entry_type: entry.entry_type.label().to_string(),
        timestamp: entry.timestamp,
        gpu_uuid: entry.gpu_uuid.clone(),
        data_preview: truncated,
    }
}
