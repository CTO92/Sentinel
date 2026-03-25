//! TMR voting logic for determining consensus from triple-redundant computations.
//!
//! Implements majority voting (2-of-3) and unanimous voting for TMR canary
//! results. The voter compares output fingerprints (SHA-256 hashes) and
//! determines which GPU(s), if any, produced a differing output.

use std::collections::HashMap;

/// Result of a TMR vote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoteResult {
    /// All three GPUs produced the same output.
    Unanimous {
        consensus: Vec<u8>,
    },
    /// Two GPUs agreed, one dissented.
    MajorityWithDissent {
        consensus: Vec<u8>,
        dissenter_index: usize,
    },
    /// All three GPUs produced different outputs (no consensus possible).
    NoConsensus,
    /// Insufficient results to determine consensus (fewer than 3 results).
    Insufficient,
}

/// Perform a TMR majority vote on output fingerprints.
///
/// `fingerprints` should contain exactly 3 entries. Each entry is a byte
/// vector representing the SHA-256 hash of the GPU's output.
///
/// Returns the vote result indicating whether consensus was reached and
/// which GPU (if any) dissented.
pub fn majority_vote(fingerprints: &[Vec<u8>]) -> VoteResult {
    if fingerprints.len() < 3 {
        return VoteResult::Insufficient;
    }

    let a = &fingerprints[0];
    let b = &fingerprints[1];
    let c = &fingerprints[2];

    if a == b && b == c {
        VoteResult::Unanimous {
            consensus: a.clone(),
        }
    } else if a == b {
        VoteResult::MajorityWithDissent {
            consensus: a.clone(),
            dissenter_index: 2,
        }
    } else if a == c {
        VoteResult::MajorityWithDissent {
            consensus: a.clone(),
            dissenter_index: 1,
        }
    } else if b == c {
        VoteResult::MajorityWithDissent {
            consensus: b.clone(),
            dissenter_index: 0,
        }
    } else {
        VoteResult::NoConsensus
    }
}

/// Perform a weighted majority vote where each GPU has a trust-based weight.
///
/// When all three disagree, the GPU with the highest trust weight is preferred
/// as the "consensus" and the other two are marked as dissenters.
///
/// Returns the most trusted output and the indices of dissenting GPUs.
pub fn weighted_vote(
    fingerprints: &[Vec<u8>],
    weights: &[f64],
) -> (Option<Vec<u8>>, Vec<usize>) {
    if fingerprints.len() < 3 || weights.len() < 3 {
        return (None, Vec::new());
    }

    // First try standard majority vote.
    match majority_vote(fingerprints) {
        VoteResult::Unanimous { consensus } => (Some(consensus), Vec::new()),
        VoteResult::MajorityWithDissent {
            consensus,
            dissenter_index,
        } => (Some(consensus), vec![dissenter_index]),
        VoteResult::NoConsensus => {
            // All three disagree. Trust the GPU with the highest weight.
            let max_idx = weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            let dissenters: Vec<usize> = (0..3).filter(|&i| i != max_idx).collect();
            (Some(fingerprints[max_idx].clone()), dissenters)
        }
        VoteResult::Insufficient => (None, Vec::new()),
    }
}

/// Compare outputs from N GPUs and find groups of agreement.
///
/// Returns a map from output fingerprint to the list of GPU indices that
/// produced that output. Useful for extended (N > 3) redundancy checks.
pub fn find_agreement_groups(fingerprints: &[Vec<u8>]) -> HashMap<Vec<u8>, Vec<usize>> {
    let mut groups: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
    for (i, fp) in fingerprints.iter().enumerate() {
        groups.entry(fp.clone()).or_default().push(i);
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unanimous_vote() {
        let fp = vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 3]];
        let result = majority_vote(&fp);
        assert_eq!(
            result,
            VoteResult::Unanimous {
                consensus: vec![1, 2, 3]
            }
        );
    }

    #[test]
    fn test_majority_with_dissent_at_index_2() {
        let fp = vec![vec![1, 2, 3], vec![1, 2, 3], vec![4, 5, 6]];
        let result = majority_vote(&fp);
        assert_eq!(
            result,
            VoteResult::MajorityWithDissent {
                consensus: vec![1, 2, 3],
                dissenter_index: 2,
            }
        );
    }

    #[test]
    fn test_majority_with_dissent_at_index_1() {
        let fp = vec![vec![1, 2, 3], vec![4, 5, 6], vec![1, 2, 3]];
        let result = majority_vote(&fp);
        assert_eq!(
            result,
            VoteResult::MajorityWithDissent {
                consensus: vec![1, 2, 3],
                dissenter_index: 1,
            }
        );
    }

    #[test]
    fn test_majority_with_dissent_at_index_0() {
        let fp = vec![vec![4, 5, 6], vec![1, 2, 3], vec![1, 2, 3]];
        let result = majority_vote(&fp);
        assert_eq!(
            result,
            VoteResult::MajorityWithDissent {
                consensus: vec![1, 2, 3],
                dissenter_index: 0,
            }
        );
    }

    #[test]
    fn test_no_consensus() {
        let fp = vec![vec![1], vec![2], vec![3]];
        let result = majority_vote(&fp);
        assert_eq!(result, VoteResult::NoConsensus);
    }

    #[test]
    fn test_insufficient() {
        let fp = vec![vec![1], vec![2]];
        let result = majority_vote(&fp);
        assert_eq!(result, VoteResult::Insufficient);
    }

    #[test]
    fn test_weighted_vote_no_consensus() {
        let fp = vec![vec![1], vec![2], vec![3]];
        let weights = vec![0.9, 0.8, 0.7];
        let (consensus, dissenters) = weighted_vote(&fp, &weights);
        assert_eq!(consensus, Some(vec![1])); // GPU 0 has highest weight.
        assert_eq!(dissenters, vec![1, 2]);
    }

    #[test]
    fn test_agreement_groups() {
        let fp = vec![vec![1], vec![2], vec![1], vec![2], vec![3]];
        let groups = find_agreement_groups(&fp);
        assert_eq!(groups.get(&vec![1u8]).unwrap(), &vec![0, 2]);
        assert_eq!(groups.get(&vec![2u8]).unwrap(), &vec![1, 3]);
        assert_eq!(groups.get(&vec![3u8]).unwrap(), &vec![4]);
    }
}
