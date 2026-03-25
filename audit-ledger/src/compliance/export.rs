//! Export audit trails in CSV, JSON, and PDF-ready (Tera template) formats.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::PgPool;

use crate::ledger::entry::{AuditEntry, AuditEntryType};
use crate::storage::postgres::{self, AuditQuery};
use crate::util::crypto;

// ---------------------------------------------------------------------------
// Flat export row
// ---------------------------------------------------------------------------

/// A flattened representation of an audit entry suitable for CSV / JSON export.
#[derive(Debug, Clone, Serialize)]
pub struct ExportRow {
    pub entry_id: u64,
    pub entry_type: String,
    pub timestamp: String,
    pub gpu_uuid: String,
    pub sm_id: String,
    pub data_base64: String,
    pub previous_hash_hex: String,
    pub entry_hash_hex: String,
    pub merkle_root_hex: String,
    pub batch_sequence: u64,
}

impl From<&AuditEntry> for ExportRow {
    fn from(e: &AuditEntry) -> Self {
        use base64::Engine;
        Self {
            entry_id: e.entry_id,
            entry_type: e.entry_type.label().to_string(),
            timestamp: e.timestamp.to_rfc3339(),
            gpu_uuid: e.gpu_uuid.clone().unwrap_or_default(),
            sm_id: e.sm_id.map(|v| v.to_string()).unwrap_or_default(),
            data_base64: base64::engine::general_purpose::STANDARD.encode(&e.data),
            previous_hash_hex: crypto::to_hex(&e.previous_hash),
            entry_hash_hex: crypto::to_hex(&e.entry_hash),
            merkle_root_hex: e
                .merkle_root
                .as_ref()
                .map(|h| crypto::to_hex(h))
                .unwrap_or_default(),
            batch_sequence: e.batch_sequence,
        }
    }
}

// ---------------------------------------------------------------------------
// Export filters
// ---------------------------------------------------------------------------

/// Parameters controlling what to export.
#[derive(Debug, Clone, Default)]
pub struct ExportFilter {
    pub gpu_uuid: Option<String>,
    pub entry_type: Option<AuditEntryType>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
}

impl From<ExportFilter> for AuditQuery {
    fn from(f: ExportFilter) -> Self {
        AuditQuery {
            gpu_uuid: f.gpu_uuid,
            entry_type: f.entry_type,
            start_time: f.start_time,
            end_time: f.end_time,
            limit: None,
            offset: None,
        }
    }
}

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------

/// Export filtered audit entries as CSV bytes.
pub async fn export_csv(pool: &PgPool, filter: ExportFilter) -> Result<Vec<u8>> {
    let entries = postgres::query_entries(pool, &filter.into())
        .await
        .context("querying entries for CSV export")?;

    let mut wtr = csv::Writer::from_writer(Vec::new());

    for entry in &entries {
        let row = ExportRow::from(entry);
        wtr.serialize(&row).context("serialising CSV row")?;
    }

    wtr.flush().context("flushing CSV writer")?;
    let data = wtr
        .into_inner()
        .map_err(|e| anyhow::anyhow!("CSV writer error: {e}"))?;
    Ok(data)
}

// ---------------------------------------------------------------------------
// JSON export
// ---------------------------------------------------------------------------

/// Export filtered audit entries as a pretty-printed JSON array.
pub async fn export_json(pool: &PgPool, filter: ExportFilter) -> Result<Vec<u8>> {
    let entries = postgres::query_entries(pool, &filter.into())
        .await
        .context("querying entries for JSON export")?;

    let rows: Vec<ExportRow> = entries.iter().map(ExportRow::from).collect();
    let json = serde_json::to_vec_pretty(&rows).context("serialising JSON")?;
    Ok(json)
}

// ---------------------------------------------------------------------------
// Tera (PDF-ready) export
// ---------------------------------------------------------------------------

/// Default Tera template for a PDF-ready HTML report.
const REPORT_TEMPLATE: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SENTINEL Audit Trail Export</title>
  <style>
    body { font-family: "Helvetica Neue", Arial, sans-serif; margin: 2em; }
    h1 { border-bottom: 2px solid #333; padding-bottom: 0.3em; }
    table { border-collapse: collapse; width: 100%; margin-top: 1em; }
    th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 0.85em; }
    th { background: #f4f4f4; }
    .meta { color: #666; font-size: 0.9em; }
  </style>
</head>
<body>
  <h1>SENTINEL Audit Trail</h1>
  <p class="meta">Generated: {{ generated_at }} | Entries: {{ entry_count }}</p>
  {% if filter_gpu %}
  <p class="meta">GPU filter: {{ filter_gpu }}</p>
  {% endif %}
  {% if filter_type %}
  <p class="meta">Type filter: {{ filter_type }}</p>
  {% endif %}
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Type</th><th>Timestamp</th><th>GPU</th>
        <th>Hash</th><th>Batch</th>
      </tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td>{{ row.entry_id }}</td>
        <td>{{ row.entry_type }}</td>
        <td>{{ row.timestamp }}</td>
        <td>{{ row.gpu_uuid }}</td>
        <td style="font-family:monospace;font-size:0.75em">{{ row.entry_hash_hex | truncate(length=16) }}</td>
        <td>{{ row.batch_sequence }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</body>
</html>
"#;

/// Export filtered audit entries as an HTML report (suitable for PDF conversion).
pub async fn export_html(pool: &PgPool, filter: ExportFilter) -> Result<Vec<u8>> {
    let filter_gpu = filter.gpu_uuid.clone().unwrap_or_default();
    let filter_type = filter
        .entry_type
        .map(|t| t.label().to_string())
        .unwrap_or_default();

    let entries = postgres::query_entries(pool, &filter.into())
        .await
        .context("querying entries for HTML export")?;

    let rows: Vec<ExportRow> = entries.iter().map(ExportRow::from).collect();

    let mut tera = tera::Tera::default();
    tera.add_raw_template("report.html", REPORT_TEMPLATE)
        .context("compiling Tera template")?;

    let mut ctx = tera::Context::new();
    ctx.insert("generated_at", &Utc::now().to_rfc3339());
    ctx.insert("entry_count", &rows.len());
    ctx.insert("filter_gpu", &filter_gpu);
    ctx.insert("filter_type", &filter_type);
    ctx.insert("rows", &rows);

    let html = tera.render("report.html", &ctx).context("rendering HTML")?;
    Ok(html.into_bytes())
}
