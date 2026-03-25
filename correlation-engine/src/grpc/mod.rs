//! gRPC service implementations for the Correlation Engine.
//!
//! Provides server implementations for:
//! - ProbeService: Bidirectional streaming for probe result ingestion
//! - AnomalyService: Bidirectional streaming for anomaly event ingestion
//! - CorrelationService: Query API for GPU and fleet health
//! - ConfigService: Dynamic configuration management

pub mod server;
pub mod probe_service;
pub mod anomaly_service;
pub mod query_service;
pub mod config_service;
