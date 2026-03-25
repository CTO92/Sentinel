//! Redis-backed state persistence.
//!
//! Persists GPU health states, Bayesian beliefs, and trust graph edges to Redis
//! so that the correlation engine can recover its state after a restart.
//!
//! Key layout:
//! - `sentinel:gpu:{uuid}:state` - GPU quarantine state record (JSON)
//! - `sentinel:gpu:{uuid}:belief` - Bayesian belief (JSON)
//! - `sentinel:trust:{pair_key}` - Trust edge (JSON)
//! - `sentinel:gpu_set` - Set of all known GPU UUIDs

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use tracing::{debug, error, info, warn};

use crate::correlation::bayesian_attribution::BayesianBelief;
use crate::health::quarantine::GpuStateRecord;
use crate::trust::trust_graph::TrustEdge;
use crate::util::config::RedisConfig;

/// Redis-backed state store.
pub struct StateStore {
    /// Redis connection manager for pooled async access.
    client: Option<redis::Client>,

    /// Key prefix for all Sentinel keys.
    prefix: String,

    /// Fallback in-memory store when Redis is not available.
    fallback: parking_lot::RwLock<InMemoryStateStore>,
}

/// In-memory fallback state store.
#[derive(Debug, Default)]
struct InMemoryStateStore {
    gpu_states: HashMap<String, String>,
    gpu_beliefs: HashMap<String, String>,
    trust_edges: HashMap<String, String>,
    gpu_set: std::collections::HashSet<String>,
}

impl StateStore {
    /// Create a new state store. Attempts to connect to Redis; falls back to
    /// in-memory storage if Redis is unavailable.
    pub async fn new(config: &RedisConfig) -> Self {
        let client = match redis::Client::open(config.url.as_str()) {
            Ok(client) => {
                // Test the connection.
                match client.get_multiplexed_async_connection().await {
                    Ok(_) => {
                        info!(url = %config.url, "Connected to Redis");
                        Some(client)
                    }
                    Err(e) => {
                        warn!(
                            url = %config.url,
                            error = %e,
                            "Failed to connect to Redis; using in-memory state store"
                        );
                        None
                    }
                }
            }
            Err(e) => {
                warn!(
                    url = %config.url,
                    error = %e,
                    "Failed to create Redis client; using in-memory state store"
                );
                None
            }
        };

        Self {
            client,
            prefix: config.key_prefix.clone(),
            fallback: parking_lot::RwLock::new(InMemoryStateStore::default()),
        }
    }

    /// Create a state store with only in-memory backend (for testing).
    pub fn in_memory(prefix: &str) -> Self {
        Self {
            client: None,
            prefix: prefix.to_string(),
            fallback: parking_lot::RwLock::new(InMemoryStateStore::default()),
        }
    }

    /// Save a GPU quarantine state record.
    pub async fn save_gpu_state(&self, gpu_uuid: &str, record: &GpuStateRecord) -> Result<()> {
        let key = format!("{}gpu:{}:state", self.prefix, gpu_uuid);
        let value = serde_json::to_string(record)?;

        if let Some(ref client) = self.client {
            let mut conn = client.get_multiplexed_async_connection().await?;
            redis::cmd("SET")
                .arg(&key)
                .arg(&value)
                .query_async::<_, ()>(&mut conn)
                .await
                .context("failed to save GPU state to Redis")?;

            // Add to GPU set.
            let set_key = format!("{}gpu_set", self.prefix);
            redis::cmd("SADD")
                .arg(&set_key)
                .arg(gpu_uuid)
                .query_async::<_, ()>(&mut conn)
                .await
                .context("failed to add GPU to set")?;
        } else {
            let mut store = self.fallback.write();
            store.gpu_states.insert(key, value);
            store.gpu_set.insert(gpu_uuid.to_string());
        }

        Ok(())
    }

    /// Load a GPU quarantine state record.
    pub async fn load_gpu_state(&self, gpu_uuid: &str) -> Result<Option<GpuStateRecord>> {
        let key = format!("{}gpu:{}:state", self.prefix, gpu_uuid);
        self.load_json(&key).await
    }

    /// Save a Bayesian belief for a GPU.
    pub async fn save_gpu_belief(&self, gpu_uuid: &str, belief: &BayesianBelief) -> Result<()> {
        let key = format!("{}gpu:{}:belief", self.prefix, gpu_uuid);
        let value = serde_json::to_string(belief)?;

        if let Some(ref client) = self.client {
            let mut conn = client.get_multiplexed_async_connection().await?;
            redis::cmd("SET")
                .arg(&key)
                .arg(&value)
                .query_async::<_, ()>(&mut conn)
                .await
                .context("failed to save GPU belief to Redis")?;
        } else {
            let mut store = self.fallback.write();
            store.gpu_beliefs.insert(key, value);
        }

        Ok(())
    }

    /// Load a Bayesian belief for a GPU.
    pub async fn load_gpu_belief(&self, gpu_uuid: &str) -> Result<Option<BayesianBelief>> {
        let key = format!("{}gpu:{}:belief", self.prefix, gpu_uuid);
        self.load_json(&key).await
    }

    /// Save a trust edge.
    pub async fn save_trust_edge(
        &self,
        gpu_a: &str,
        gpu_b: &str,
        edge: &TrustEdge,
    ) -> Result<()> {
        let pair_key = if gpu_a <= gpu_b {
            format!("{}:{}",gpu_a, gpu_b)
        } else {
            format!("{}:{}", gpu_b, gpu_a)
        };
        let key = format!("{}trust:{}", self.prefix, pair_key);
        let value = serde_json::to_string(edge)?;

        if let Some(ref client) = self.client {
            let mut conn = client.get_multiplexed_async_connection().await?;
            redis::cmd("SET")
                .arg(&key)
                .arg(&value)
                .query_async::<_, ()>(&mut conn)
                .await
                .context("failed to save trust edge to Redis")?;
        } else {
            let mut store = self.fallback.write();
            store.trust_edges.insert(key, value);
        }

        Ok(())
    }

    /// Load a trust edge.
    pub async fn load_trust_edge(
        &self,
        gpu_a: &str,
        gpu_b: &str,
    ) -> Result<Option<TrustEdge>> {
        let pair_key = if gpu_a <= gpu_b {
            format!("{}:{}", gpu_a, gpu_b)
        } else {
            format!("{}:{}", gpu_b, gpu_a)
        };
        let key = format!("{}trust:{}", self.prefix, pair_key);
        self.load_json(&key).await
    }

    /// Get all known GPU UUIDs from the store.
    pub async fn load_all_gpu_uuids(&self) -> Result<Vec<String>> {
        if let Some(ref client) = self.client {
            let mut conn = client.get_multiplexed_async_connection().await?;
            let set_key = format!("{}gpu_set", self.prefix);
            let members: Vec<String> = redis::cmd("SMEMBERS")
                .arg(&set_key)
                .query_async(&mut conn)
                .await
                .context("failed to load GPU set from Redis")?;
            Ok(members)
        } else {
            let store = self.fallback.read();
            Ok(store.gpu_set.iter().cloned().collect())
        }
    }

    /// Load all GPU states from the store (for startup recovery).
    pub async fn load_all_gpu_states(&self) -> Result<HashMap<String, GpuStateRecord>> {
        let uuids = self.load_all_gpu_uuids().await?;
        let mut states = HashMap::new();

        for uuid in uuids {
            if let Some(record) = self.load_gpu_state(&uuid).await? {
                states.insert(uuid, record);
            }
        }

        Ok(states)
    }

    /// Load all GPU beliefs from the store (for startup recovery).
    pub async fn load_all_gpu_beliefs(&self) -> Result<HashMap<String, BayesianBelief>> {
        let uuids = self.load_all_gpu_uuids().await?;
        let mut beliefs = HashMap::new();

        for uuid in uuids {
            if let Some(belief) = self.load_gpu_belief(&uuid).await? {
                beliefs.insert(uuid, belief);
            }
        }

        Ok(beliefs)
    }

    /// Check if Redis is connected.
    pub fn is_redis_connected(&self) -> bool {
        self.client.is_some()
    }

    /// Generic JSON load helper.
    async fn load_json<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        if let Some(ref client) = self.client {
            let mut conn = client.get_multiplexed_async_connection().await?;
            let value: Option<String> = redis::cmd("GET")
                .arg(key)
                .query_async(&mut conn)
                .await
                .context("failed to load from Redis")?;

            match value {
                Some(json) => Ok(Some(serde_json::from_str(&json)?)),
                None => Ok(None),
            }
        } else {
            let store = self.fallback.read();
            // Check all in-memory maps.
            let value = store
                .gpu_states
                .get(key)
                .or_else(|| store.gpu_beliefs.get(key))
                .or_else(|| store.trust_edges.get(key));

            match value {
                Some(json) => Ok(Some(serde_json::from_str(json)?)),
                None => Ok(None),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::health::quarantine::QuarantineState;

    #[tokio::test]
    async fn test_in_memory_state_store() {
        let store = StateStore::in_memory("test:");

        // Save and load GPU state.
        let record = GpuStateRecord {
            state: QuarantineState::Suspect,
            state_entered_at: chrono::Utc::now(),
            reason: "test".to_string(),
            deep_test_passes: 0,
            history: Vec::new(),
        };

        store.save_gpu_state("gpu-1", &record).await.unwrap();
        let loaded = store.load_gpu_state("gpu-1").await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().state, QuarantineState::Suspect);
    }

    #[tokio::test]
    async fn test_in_memory_belief_store() {
        let store = StateStore::in_memory("test:");

        let belief = BayesianBelief::new(1000.0, 5.0);
        store.save_gpu_belief("gpu-1", &belief).await.unwrap();

        let loaded = store.load_gpu_belief("gpu-1").await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().alpha, 1000.0);
    }

    #[tokio::test]
    async fn test_gpu_uuid_tracking() {
        let store = StateStore::in_memory("test:");

        let record = GpuStateRecord {
            state: QuarantineState::Healthy,
            state_entered_at: chrono::Utc::now(),
            reason: "init".to_string(),
            deep_test_passes: 0,
            history: Vec::new(),
        };

        store.save_gpu_state("gpu-1", &record).await.unwrap();
        store.save_gpu_state("gpu-2", &record).await.unwrap();

        let uuids = store.load_all_gpu_uuids().await.unwrap();
        assert_eq!(uuids.len(), 2);
    }
}
