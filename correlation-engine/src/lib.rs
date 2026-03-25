#![allow(dead_code)]

//! SENTINEL Correlation Engine library crate.
//!
//! The Correlation Engine is the central intelligence component of the SENTINEL
//! Silent Data Corruption detection framework. It aggregates events from probe
//! agents and inference/training monitors, correlates probe failures with output
//! anomalies, maintains a GPU trust graph, performs Bayesian attribution of
//! reliability, and manages GPU quarantine lifecycle.
//!
//! # Architecture
//!
//! The engine uses a Tokio-based actor model where each GPU has a dedicated
//! health actor that processes events sequentially, eliminating lock contention.
//! A central trust graph actor and quarantine manager actor handle cross-GPU
//! state transitions.
//!
//! # Modules
//!
//! - [`grpc`] - gRPC service implementations for event ingestion and queries
//! - [`correlation`] - Core correlation logic, Bayesian attribution, pattern matching
//! - [`trust`] - GPU trust graph and TMR scheduling
//! - [`health`] - GPU health tracking and quarantine state machine
//! - [`storage`] - Event persistence and state store (Redis)
//! - [`alerting`] - Alert routing and integrations (PagerDuty, Slack, webhooks)
//! - [`util`] - Configuration, metrics, and shared utilities

pub mod grpc;
pub mod correlation;
pub mod trust;
pub mod health;
pub mod storage;
pub mod alerting;
pub mod util;

/// Re-export of generated protobuf types. These are produced by `tonic-build`
/// at compile time from the proto definitions in `../proto/sentinel/v1/`.
///
/// The generated code includes Rust structs for all protobuf messages and
/// tonic server/client trait implementations for all gRPC services.
#[allow(clippy::all, clippy::pedantic, missing_docs)]
pub mod proto {
    pub mod sentinel {
        pub mod v1 {
            tonic::include_proto!("sentinel.v1");
        }
    }
}
