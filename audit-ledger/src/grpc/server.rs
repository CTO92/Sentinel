//! gRPC server bootstrap.
//!
//! Starts both the ingest and query services on a single tonic transport.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use sqlx::PgPool;
use tokio::sync::Mutex;
use tonic::transport::Server;
use tracing::info;

use crate::grpc::ingest_service::{AuditIngestService, AuditIngestServer};
use crate::grpc::query_service::{AuditQueryService, AuditQueryServer};
use crate::ledger::merkle_chain::{BatchAccumulator, ChainBuilder};

/// Shared state passed to gRPC service implementations.
pub struct SharedState {
    pub pool: PgPool,
    pub chain_builder: Mutex<ChainBuilder>,
    pub accumulator: Mutex<BatchAccumulator>,
}

/// Start the gRPC server.
pub async fn start(addr: SocketAddr, pool: PgPool, chain_builder: ChainBuilder) -> Result<()> {
    let state = Arc::new(SharedState {
        pool: pool.clone(),
        chain_builder: Mutex::new(chain_builder),
        accumulator: Mutex::new(BatchAccumulator::default_config()),
    });

    let ingest_svc = AuditIngestServer::new(AuditIngestService::new(state.clone()));
    let query_svc = AuditQueryServer::new(AuditQueryService::new(state.clone()));

    info!(%addr, "Starting gRPC server");

    Server::builder()
        .add_service(ingest_svc)
        .add_service(query_svc)
        .serve(addr)
        .await
        .map_err(|e| anyhow::anyhow!("gRPC server error: {e}"))?;

    Ok(())
}
