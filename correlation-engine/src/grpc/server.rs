//! gRPC server setup and lifecycle management.
//!
//! Configures and starts the tonic gRPC server with all service implementations
//! registered. Also starts the axum REST API server for the dashboard.

use std::sync::Arc;

use anyhow::Result;
use axum::{extract::State, routing::get, Json, Router};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::correlation::engine::EngineHandle;
use crate::health::quarantine::QuarantineManager;
use crate::trust::trust_graph::TrustGraph;
use crate::util::config::AppConfig;
use crate::util::metrics;

/// Shared application state for the REST API.
#[derive(Clone)]
pub struct AppState {
    pub engine_handle: EngineHandle,
    pub trust_graph: Arc<RwLock<TrustGraph>>,
    pub quarantine_manager: Arc<RwLock<QuarantineManager>>,
}

/// Start the REST API server for the dashboard.
pub async fn start_rest_server(config: &AppConfig, state: AppState) -> Result<()> {
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/metrics", get(prometheus_metrics))
        .route("/api/v1/gpu/:gpu_uuid", get(get_gpu_health))
        .route("/api/v1/fleet", get(get_fleet_health))
        .route("/api/v1/gpu/:gpu_uuid/history", get(get_gpu_history))
        .route("/api/v1/trust/coverage", get(get_trust_coverage))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let bind_addr = format!("{}:{}", config.rest.bind_addr, config.rest.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    info!(addr = %bind_addr, "REST API server started");

    axum::serve(listener, app).await?;
    Ok(())
}

/// Start the Prometheus metrics scrape endpoint.
pub async fn start_metrics_server(config: &AppConfig) -> Result<()> {
    let app = Router::new().route("/metrics", get(prometheus_metrics));

    let bind_addr = format!("{}:{}", config.metrics.bind_addr, config.metrics.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    info!(addr = %bind_addr, "Metrics server started");

    axum::serve(listener, app).await?;
    Ok(())
}

/// Health check endpoint.
async fn health_check() -> &'static str {
    "OK"
}

/// Prometheus metrics endpoint.
async fn prometheus_metrics() -> String {
    metrics::gather_metrics()
}

/// GET /api/v1/gpu/:gpu_uuid - Get GPU health.
async fn get_gpu_health(
    State(state): State<AppState>,
    axum::extract::Path(gpu_uuid): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, axum::http::StatusCode> {
    match state.engine_handle.query_gpu_health(gpu_uuid).await {
        Ok(Some(snapshot)) => Ok(Json(serde_json::json!({
            "gpu_uuid": snapshot.gpu_uuid,
            "state": format!("{}", snapshot.state),
            "reliability_score": snapshot.reliability_score,
            "alpha": snapshot.alpha,
            "beta": snapshot.beta,
            "probe_pass_count": snapshot.probe_pass_count,
            "probe_fail_count": snapshot.probe_fail_count,
            "anomaly_count": snapshot.anomaly_count,
            "recent_patterns": snapshot.recent_patterns.len(),
        }))),
        Ok(None) => Err(axum::http::StatusCode::NOT_FOUND),
        Err(_) => Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// GET /api/v1/fleet - Get fleet health summary.
async fn get_fleet_health(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, axum::http::StatusCode> {
    match state.engine_handle.query_fleet_health().await {
        Ok(snapshot) => Ok(Json(serde_json::json!({
            "total_gpus": snapshot.total_gpus,
            "healthy": snapshot.healthy,
            "suspect": snapshot.suspect,
            "quarantined": snapshot.quarantined,
            "deep_test": snapshot.deep_test,
            "condemned": snapshot.condemned,
            "average_reliability": snapshot.average_reliability,
        }))),
        Err(_) => Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// GET /api/v1/gpu/:gpu_uuid/history - Get GPU event history.
async fn get_gpu_history(
    State(state): State<AppState>,
    axum::extract::Path(gpu_uuid): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, axum::http::StatusCode> {
    match state.engine_handle.query_gpu_history(gpu_uuid, 100).await {
        Ok(events) => {
            let event_summaries: Vec<serde_json::Value> = events
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "event_id": e.event_id,
                        "event_type": format!("{}", e.event_type),
                        "timestamp": e.timestamp.to_rfc3339(),
                        "severity": e.severity,
                        "score": e.score,
                    })
                })
                .collect();

            Ok(Json(serde_json::json!({
                "events": event_summaries,
                "count": event_summaries.len(),
            })))
        }
        Err(_) => Err(axum::http::StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// GET /api/v1/trust/coverage - Get trust graph coverage.
async fn get_trust_coverage(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, axum::http::StatusCode> {
    let trust = state.trust_graph.read().await;
    Ok(Json(serde_json::json!({
        "coverage_percent": trust.coverage_percent(),
        "gpu_count": trust.gpu_count(),
        "edge_count": trust.edge_count(),
        "min_trust_score": trust.min_trust_score(),
        "mean_trust_score": trust.mean_trust_score(),
    })))
}
