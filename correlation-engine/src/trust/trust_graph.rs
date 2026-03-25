//! GPU trust graph for cross-validation tracking.
//!
//! The trust graph is an N x N sparse matrix represented as an adjacency list.
//! Each edge T[i][j] tracks the number of successful and failed TMR
//! cross-validations between GPUs i and j, yielding a pairwise trust score.
//!
//! Trust scores are used for:
//! - TMR partner selection (prefer high-trust pairs)
//! - Transitive trust chains (if A trusts B and B trusts C, A has indirect trust in C)
//! - Coverage tracking (what percentage of GPU pairs have been compared)

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::util::config::TrustConfig;

/// An edge in the trust graph representing pairwise comparison history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustEdge {
    /// Number of times these GPUs produced matching outputs.
    pub agreement_count: u64,

    /// Number of times these GPUs produced differing outputs.
    pub disagreement_count: u64,

    /// Trust score: agreement / (agreement + disagreement).
    pub trust_score: f64,

    /// Timestamp of the most recent comparison.
    pub last_comparison: DateTime<Utc>,
}

impl TrustEdge {
    /// Create a new edge with zero history.
    fn new() -> Self {
        Self {
            agreement_count: 0,
            disagreement_count: 0,
            trust_score: 0.0,
            last_comparison: Utc::now(),
        }
    }

    /// Total number of comparisons.
    pub fn total_comparisons(&self) -> u64 {
        self.agreement_count + self.disagreement_count
    }

    /// Recalculate the trust score.
    fn recalculate(&mut self) {
        let total = self.total_comparisons();
        if total == 0 {
            self.trust_score = 0.0;
        } else {
            self.trust_score = self.agreement_count as f64 / total as f64;
        }
    }
}

/// A canonical pair key that is order-independent: (min(a,b), max(a,b)).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct PairKey(String, String);

impl PairKey {
    fn new(a: &str, b: &str) -> Self {
        if a <= b {
            PairKey(a.to_string(), b.to_string())
        } else {
            PairKey(b.to_string(), a.to_string())
        }
    }
}

/// The trust graph tracking pairwise GPU cross-validation history.
#[derive(Debug)]
pub struct TrustGraph {
    /// Configuration.
    config: TrustConfig,

    /// All known GPU UUIDs.
    gpus: HashSet<String>,

    /// Pairwise trust edges.
    edges: HashMap<PairKey, TrustEdge>,
}

impl TrustGraph {
    /// Create a new empty trust graph.
    pub fn new(config: TrustConfig) -> Self {
        Self {
            config,
            gpus: HashSet::new(),
            edges: HashMap::new(),
        }
    }

    /// Register a GPU in the trust graph.
    pub fn register_gpu(&mut self, gpu_uuid: &str) {
        self.gpus.insert(gpu_uuid.to_string());
    }

    /// Record an agreement between two GPUs (they produced matching outputs).
    pub fn record_agreement(&mut self, gpu_a: &str, gpu_b: &str) {
        self.gpus.insert(gpu_a.to_string());
        self.gpus.insert(gpu_b.to_string());

        let key = PairKey::new(gpu_a, gpu_b);
        let edge = self.edges.entry(key).or_insert_with(TrustEdge::new);
        edge.agreement_count += 1;
        edge.last_comparison = Utc::now();
        edge.recalculate();
    }

    /// Record a disagreement between two GPUs (they produced differing outputs).
    pub fn record_disagreement(&mut self, gpu_a: &str, gpu_b: &str) {
        self.gpus.insert(gpu_a.to_string());
        self.gpus.insert(gpu_b.to_string());

        let key = PairKey::new(gpu_a, gpu_b);
        let edge = self.edges.entry(key).or_insert_with(TrustEdge::new);
        edge.disagreement_count += 1;
        edge.last_comparison = Utc::now();
        edge.recalculate();
    }

    /// Get the trust edge between two GPUs.
    pub fn get_edge(&self, gpu_a: &str, gpu_b: &str) -> Option<&TrustEdge> {
        let key = PairKey::new(gpu_a, gpu_b);
        self.edges.get(&key)
    }

    /// Get the trust score between two GPUs.
    /// Returns `None` if the pair has never been compared.
    pub fn trust_score(&self, gpu_a: &str, gpu_b: &str) -> Option<f64> {
        self.get_edge(gpu_a, gpu_b).map(|e| e.trust_score)
    }

    /// Compute the coverage percentage: fraction of all possible pairs that
    /// have been compared at least `min_comparisons` times.
    pub fn coverage_percent(&self) -> f64 {
        let n = self.gpus.len();
        if n < 2 {
            return 100.0;
        }

        let total_possible = n * (n - 1) / 2;
        let covered = self
            .edges
            .values()
            .filter(|e| e.total_comparisons() >= self.config.min_comparisons)
            .count();

        (covered as f64 / total_possible as f64) * 100.0
    }

    /// Get the number of GPUs in the graph.
    pub fn gpu_count(&self) -> usize {
        self.gpus.len()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Select the best TMR partner pair for a target GPU.
    ///
    /// Returns two GPUs that:
    /// 1. Are not the target GPU
    /// 2. Have high mutual trust with each other
    /// 3. Have reasonable trust with the target
    /// 4. Prefer pairs that have been least-recently compared with the target
    ///    (to maximize coverage)
    pub fn select_tmr_partners(
        &self,
        target_gpu: &str,
        available_gpus: &[String],
    ) -> Option<(String, String)> {
        if available_gpus.len() < 2 {
            return None;
        }

        // Filter out the target GPU.
        let candidates: Vec<&String> = available_gpus
            .iter()
            .filter(|g| g.as_str() != target_gpu)
            .collect();

        if candidates.len() < 2 {
            return None;
        }

        // Score each pair of candidates.
        let mut best_pair: Option<(String, String)> = None;
        let mut best_score = f64::NEG_INFINITY;

        for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                let a = candidates[i];
                let b = candidates[j];

                // Mutual trust between the two partners.
                let mutual_trust = self
                    .trust_score(a, b)
                    .unwrap_or(0.5); // Unknown pairs get neutral score.

                // Trust between target and each partner.
                let target_a_trust = self.trust_score(target_gpu, a).unwrap_or(0.5);
                let target_b_trust = self.trust_score(target_gpu, b).unwrap_or(0.5);

                // Coverage bonus: prefer pairs that haven't been compared with
                // the target recently (to increase graph coverage).
                let coverage_bonus_a = self
                    .get_edge(target_gpu, a)
                    .map(|e| 1.0 / (1.0 + e.total_comparisons() as f64))
                    .unwrap_or(2.0); // Never compared = highest bonus.

                let coverage_bonus_b = self
                    .get_edge(target_gpu, b)
                    .map(|e| 1.0 / (1.0 + e.total_comparisons() as f64))
                    .unwrap_or(2.0);

                let score = mutual_trust * 0.4
                    + (target_a_trust + target_b_trust) * 0.2
                    + (coverage_bonus_a + coverage_bonus_b) * 0.1;

                if score > best_score {
                    best_score = score;
                    best_pair = Some((a.clone(), b.clone()));
                }
            }
        }

        best_pair
    }

    /// Compute transitive trust between two GPUs that may not have been
    /// directly compared.
    ///
    /// Uses a BFS-like approach to find trust paths and returns the maximum
    /// trust achievable through intermediate nodes. Transitive trust decays
    /// multiplicatively: trust(A->C) via B = trust(A->B) * trust(B->C).
    pub fn transitive_trust(&self, gpu_a: &str, gpu_b: &str, max_hops: usize) -> f64 {
        // Direct trust.
        if let Some(direct) = self.trust_score(gpu_a, gpu_b) {
            return direct;
        }

        if max_hops == 0 {
            return 0.0;
        }

        // BFS for transitive trust.
        let mut best_trust = 0.0f64;
        let mut visited = HashSet::new();
        visited.insert(gpu_a.to_string());

        // (current_node, accumulated_trust, depth)
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((gpu_a.to_string(), 1.0f64, 0usize));

        while let Some((current, acc_trust, depth)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }

            // Find all neighbors of current.
            for gpu in &self.gpus {
                if visited.contains(gpu.as_str()) {
                    continue;
                }

                if let Some(edge) = self.get_edge(&current, gpu) {
                    if edge.total_comparisons() < self.config.min_comparisons {
                        continue;
                    }

                    let path_trust = acc_trust * edge.trust_score;

                    if gpu == gpu_b {
                        best_trust = best_trust.max(path_trust);
                    } else {
                        visited.insert(gpu.clone());
                        queue.push_back((gpu.clone(), path_trust, depth + 1));
                    }
                }
            }
        }

        best_trust
    }

    /// Get all edges as a list (for serialization / snapshot).
    pub fn all_edges(&self) -> Vec<(String, String, TrustEdge)> {
        self.edges
            .iter()
            .map(|(key, edge)| (key.0.clone(), key.1.clone(), edge.clone()))
            .collect()
    }

    /// Get the minimum trust score across all edges (with sufficient comparisons).
    pub fn min_trust_score(&self) -> f64 {
        self.edges
            .values()
            .filter(|e| e.total_comparisons() >= self.config.min_comparisons)
            .map(|e| e.trust_score)
            .fold(f64::MAX, f64::min)
    }

    /// Get the mean trust score across all edges (with sufficient comparisons).
    pub fn mean_trust_score(&self) -> f64 {
        let valid_edges: Vec<f64> = self
            .edges
            .values()
            .filter(|e| e.total_comparisons() >= self.config.min_comparisons)
            .map(|e| e.trust_score)
            .collect();

        if valid_edges.is_empty() {
            return 0.0;
        }

        valid_edges.iter().sum::<f64>() / valid_edges.len() as f64
    }

    /// Get all registered GPU UUIDs.
    pub fn all_gpus(&self) -> Vec<String> {
        self.gpus.iter().cloned().collect()
    }

    /// Apply time decay to all trust edges. This should be called periodically
    /// (e.g., once per hour) to gradually reduce confidence in old comparisons.
    pub fn apply_decay(&mut self) {
        let factor = self.config.decay_factor;
        for edge in self.edges.values_mut() {
            edge.agreement_count =
                (edge.agreement_count as f64 * factor).round() as u64;
            edge.disagreement_count =
                (edge.disagreement_count as f64 * factor).round() as u64;
            edge.recalculate();
        }

        // Remove edges that have decayed to zero.
        self.edges.retain(|_, e| e.total_comparisons() > 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> TrustConfig {
        TrustConfig {
            min_comparisons: 1,
            decay_factor: 0.999,
        }
    }

    #[test]
    fn test_agreement_increases_trust() {
        let mut graph = TrustGraph::new(test_config());
        graph.record_agreement("gpu-1", "gpu-2");
        assert_eq!(graph.trust_score("gpu-1", "gpu-2"), Some(1.0));
    }

    #[test]
    fn test_disagreement_decreases_trust() {
        let mut graph = TrustGraph::new(test_config());
        graph.record_agreement("gpu-1", "gpu-2");
        graph.record_disagreement("gpu-1", "gpu-2");
        assert_eq!(graph.trust_score("gpu-1", "gpu-2"), Some(0.5));
    }

    #[test]
    fn test_order_independent() {
        let mut graph = TrustGraph::new(test_config());
        graph.record_agreement("gpu-2", "gpu-1");
        assert_eq!(graph.trust_score("gpu-1", "gpu-2"), Some(1.0));
    }

    #[test]
    fn test_coverage() {
        let mut graph = TrustGraph::new(test_config());
        graph.register_gpu("gpu-1");
        graph.register_gpu("gpu-2");
        graph.register_gpu("gpu-3");

        // 3 GPUs = 3 possible pairs. 0 compared.
        assert_eq!(graph.coverage_percent(), 0.0);

        graph.record_agreement("gpu-1", "gpu-2");
        // 1 of 3 pairs compared.
        let coverage = graph.coverage_percent();
        assert!((coverage - 33.333).abs() < 1.0);
    }

    #[test]
    fn test_partner_selection() {
        let mut graph = TrustGraph::new(test_config());
        for i in 1..=5 {
            graph.register_gpu(&format!("gpu-{}", i));
        }

        // Build some trust history.
        for _ in 0..10 {
            graph.record_agreement("gpu-2", "gpu-3");
        }

        let available: Vec<String> = (1..=5).map(|i| format!("gpu-{}", i)).collect();
        let partners = graph.select_tmr_partners("gpu-1", &available);
        assert!(partners.is_some());
    }

    #[test]
    fn test_transitive_trust() {
        let mut graph = TrustGraph::new(test_config());
        // A trusts B, B trusts C, but A and C have never met.
        for _ in 0..10 {
            graph.record_agreement("gpu-a", "gpu-b");
            graph.record_agreement("gpu-b", "gpu-c");
        }

        let trust = graph.transitive_trust("gpu-a", "gpu-c", 2);
        assert!(trust > 0.0);
        // Trust should be <= min(trust(A,B), trust(B,C)) = 1.0 * 1.0 = 1.0
        assert!(trust <= 1.0);
    }

    #[test]
    fn test_decay() {
        let mut graph = TrustGraph::new(TrustConfig {
            min_comparisons: 1,
            decay_factor: 0.5,
        });
        graph.record_agreement("gpu-1", "gpu-2");
        graph.record_agreement("gpu-1", "gpu-2");
        // 2 agreements.

        graph.apply_decay();
        let edge = graph.get_edge("gpu-1", "gpu-2").unwrap();
        assert_eq!(edge.agreement_count, 1); // 2 * 0.5 = 1
    }
}
