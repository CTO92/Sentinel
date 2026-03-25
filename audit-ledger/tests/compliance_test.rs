//! Tests for compliance report data structures and export formatting.
//!
//! These tests exercise the report structures and export logic without
//! requiring a live PostgreSQL connection.

use sentinel_audit_ledger::compliance::export::ExportRow;
use sentinel_audit_ledger::ledger::entry::{AuditEntry, AuditEntryType};
use sentinel_audit_ledger::util::crypto::ZERO_HASH;
use chrono::Utc;

fn sample_entry(id: u64, entry_type: AuditEntryType, gpu: Option<&str>) -> AuditEntry {
    let data = format!("{{\"id\": {id}}}").into_bytes();
    let hash = AuditEntry::compute_hash(&data, &ZERO_HASH);
    AuditEntry {
        entry_id: id,
        entry_type,
        timestamp: Utc::now(),
        gpu_uuid: gpu.map(String::from),
        sm_id: None,
        data,
        previous_hash: ZERO_HASH,
        entry_hash: hash,
        merkle_root: None,
        batch_sequence: 1,
    }
}

#[test]
fn export_row_from_entry() {
    let entry = sample_entry(1, AuditEntryType::ProbeResult, Some("GPU-ABC"));
    let row = ExportRow::from(&entry);

    assert_eq!(row.entry_id, 1);
    assert_eq!(row.entry_type, "ProbeResult");
    assert_eq!(row.gpu_uuid, "GPU-ABC");
    assert!(!row.entry_hash_hex.is_empty());
    assert!(!row.data_base64.is_empty());
    assert_eq!(row.batch_sequence, 1);
}

#[test]
fn export_row_empty_gpu() {
    let entry = sample_entry(2, AuditEntryType::SystemEvent, None);
    let row = ExportRow::from(&entry);
    assert_eq!(row.gpu_uuid, "");
}

#[test]
fn export_row_all_types() {
    let types = [
        AuditEntryType::ProbeResult,
        AuditEntryType::AnomalyEvent,
        AuditEntryType::QuarantineAction,
        AuditEntryType::ConfigChange,
        AuditEntryType::TmrResult,
        AuditEntryType::SystemEvent,
    ];

    for t in &types {
        let entry = sample_entry(0, *t, None);
        let row = ExportRow::from(&entry);
        assert_eq!(row.entry_type, t.label());
    }
}

#[test]
fn export_row_serializes_to_json() {
    let entry = sample_entry(42, AuditEntryType::AnomalyEvent, Some("GPU-XYZ"));
    let row = ExportRow::from(&entry);

    let json = serde_json::to_string(&row).expect("serialize to JSON");
    assert!(json.contains("\"entry_id\":42"));
    assert!(json.contains("\"entry_type\":\"AnomalyEvent\""));
    assert!(json.contains("\"gpu_uuid\":\"GPU-XYZ\""));
}

#[test]
fn export_row_csv_roundtrip() {
    let entry = sample_entry(10, AuditEntryType::ConfigChange, Some("GPU-001"));
    let row = ExportRow::from(&entry);

    let mut wtr = csv::Writer::from_writer(Vec::new());
    wtr.serialize(&row).expect("serialize CSV row");
    wtr.flush().expect("flush");
    let csv_bytes = wtr.into_inner().expect("into_inner");
    let csv_str = String::from_utf8(csv_bytes).expect("UTF-8");

    // Should contain the header and one data row.
    let lines: Vec<&str> = csv_str.lines().collect();
    assert_eq!(lines.len(), 2); // header + data
    assert!(lines[0].contains("entry_id"));
    assert!(lines[1].contains("10"));
}

#[test]
fn soc2_report_struct_serializes() {
    use sentinel_audit_ledger::compliance::soc2_report::*;

    let report = Soc2Report {
        metadata: ReportMetadata {
            report_type: "SOC 2 Type II".into(),
            generated_at: Utc::now(),
            period_start: Utc::now(),
            period_end: Utc::now(),
            generated_by: "test".into(),
        },
        a1_2_availability: AvailabilityControl {
            control_id: "A1.2".into(),
            control_description: "test".into(),
            total_system_events: 100,
            quarantine_events: 5,
            unquarantine_events: 3,
            gpu_availability_ratio: 0.98,
            events: vec![],
        },
        cc7_2_security_monitoring: SecurityMonitoringControl {
            control_id: "CC7.2".into(),
            control_description: "test".into(),
            total_anomalies_detected: 10,
            total_probe_results: 10000,
            anomaly_rate: 0.001,
            tmr_comparisons: 500,
            events: vec![],
        },
        cc8_1_change_management: ChangeManagementControl {
            control_id: "CC8.1".into(),
            control_description: "test".into(),
            total_config_changes: 3,
            events: vec![],
        },
        summary: ReportSummary {
            total_events_in_period: 10113,
            integrity_verified: true,
            chain_continuous: true,
        },
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize");
    assert!(json.contains("SOC 2 Type II"));
    assert!(json.contains("A1.2"));
    assert!(json.contains("CC7.2"));
    assert!(json.contains("CC8.1"));
}

#[test]
fn iso27001_report_struct_serializes() {
    use sentinel_audit_ledger::compliance::iso27001_report::*;

    let report = Iso27001Report {
        metadata: ReportMetadata {
            report_type: "ISO 27001 Internal Audit".into(),
            standard: "ISO/IEC 27001:2022".into(),
            generated_at: Utc::now(),
            period_start: Utc::now(),
            period_end: Utc::now(),
            generated_by: "test".into(),
        },
        a8_asset_management: AssetManagementControl {
            control_id: "A.8".into(),
            control_title: "Asset Management".into(),
            description: "test".into(),
            gpus_observed: vec!["GPU-A".into(), "GPU-B".into()],
            gpu_events_total: 200,
            quarantine_actions: 3,
            config_changes: 1,
            findings: vec![],
        },
        a12_operations_security: OperationsSecurityControl {
            control_id: "A.12".into(),
            control_title: "Operations Security".into(),
            description: "test".into(),
            anomalies_detected: 5,
            probes_executed: 5000,
            tmr_comparisons: 200,
            mean_time_to_detect_seconds: Some(1.5),
            incidents_count: 2,
            findings: vec![Finding {
                severity: Severity::Medium,
                description: "Elevated anomaly rate".into(),
                evidence_entry_ids: vec![101, 102],
            }],
        },
        summary: ReportSummary {
            total_events: 5205,
            chain_integrity_verified: true,
            controls_assessed: 2,
            findings_total: 1,
        },
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize");
    assert!(json.contains("ISO/IEC 27001:2022"));
    assert!(json.contains("A.8"));
    assert!(json.contains("A.12"));
    assert!(json.contains("Elevated anomaly rate"));
}
