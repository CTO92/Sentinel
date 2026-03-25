//! Cryptographic utilities for the SENTINEL Audit Ledger.
//!
//! Provides SHA-256 hashing, HMAC-SHA256 signing, Merkle-tree root computation,
//! and constant-time comparison helpers.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// Length of a SHA-256 digest in bytes.
pub const HASH_LEN: usize = 32;

/// A 32-byte SHA-256 hash.
pub type Hash256 = [u8; HASH_LEN];

/// The zero hash, used as the "previous hash" for the very first entry in a chain.
pub const ZERO_HASH: Hash256 = [0u8; HASH_LEN];

// ---------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------

/// Compute the SHA-256 digest of a single byte slice.
pub fn sha256(data: &[u8]) -> Hash256 {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; HASH_LEN];
    out.copy_from_slice(&result);
    out
}

/// Compute the chain hash for an audit entry:
///
/// ```text
/// H = SHA-256( data || previous_hash )
/// ```
pub fn chain_hash(data: &[u8], previous_hash: &Hash256) -> Hash256 {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.update(previous_hash);
    let result = hasher.finalize();
    let mut out = [0u8; HASH_LEN];
    out.copy_from_slice(&result);
    out
}

// ---------------------------------------------------------------------------
// HMAC-SHA256
// ---------------------------------------------------------------------------

type HmacSha256 = Hmac<Sha256>;

/// Compute the HMAC-SHA256 tag for `data` under the given `key`.
pub fn hmac_sha256(key: &[u8], data: &[u8]) -> Hash256 {
    let mut mac =
        HmacSha256::new_from_slice(key).expect("HMAC-SHA256 accepts any key length");
    mac.update(data);
    let result = mac.finalize().into_bytes();
    let mut out = [0u8; HASH_LEN];
    out.copy_from_slice(&result);
    out
}

/// Verify an HMAC-SHA256 tag in constant time.
pub fn hmac_sha256_verify(key: &[u8], data: &[u8], expected_tag: &[u8]) -> bool {
    let computed = hmac_sha256(key, data);
    constant_time_eq(&computed, expected_tag)
}

// ---------------------------------------------------------------------------
// Merkle tree
// ---------------------------------------------------------------------------

/// Compute the Merkle root of a list of leaf hashes.
///
/// Uses the standard binary Merkle tree construction.  If the number of nodes
/// at any level is odd, the last node is promoted (not duplicated) to avoid
/// second-preimage weaknesses.
///
/// Returns [`ZERO_HASH`] when `leaves` is empty.
pub fn merkle_root(leaves: &[Hash256]) -> Hash256 {
    if leaves.is_empty() {
        return ZERO_HASH;
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut current_level: Vec<Hash256> = leaves.to_vec();

    while current_level.len() > 1 {
        let mut next_level = Vec::with_capacity((current_level.len() + 1) / 2);

        let mut i = 0;
        while i + 1 < current_level.len() {
            let mut hasher = Sha256::new();
            hasher.update(current_level[i]);
            hasher.update(current_level[i + 1]);
            let result = hasher.finalize();
            let mut out = [0u8; HASH_LEN];
            out.copy_from_slice(&result);
            next_level.push(out);
            i += 2;
        }

        // Odd node: promote without duplication.
        if i < current_level.len() {
            next_level.push(current_level[i]);
        }

        current_level = next_level;
    }

    current_level[0]
}

// ---------------------------------------------------------------------------
// Constant-time comparison
// ---------------------------------------------------------------------------

/// Constant-time equality check for two byte slices.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}

// ---------------------------------------------------------------------------
// Hex helpers (re-exports for convenience)
// ---------------------------------------------------------------------------

/// Encode bytes as a lowercase hex string.
pub fn to_hex(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Decode a hex string to bytes.
pub fn from_hex(s: &str) -> Result<Vec<u8>, hex::FromHexError> {
    hex::decode(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256(b"");
        assert_eq!(
            to_hex(&hash),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_chain_hash_deterministic() {
        let h1 = chain_hash(b"hello", &ZERO_HASH);
        let h2 = chain_hash(b"hello", &ZERO_HASH);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_chain_hash_varies_with_previous() {
        let h1 = chain_hash(b"hello", &ZERO_HASH);
        let h2 = chain_hash(b"hello", &sha256(b"something"));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hmac_roundtrip() {
        let key = b"secret-key";
        let data = b"important data";
        let tag = hmac_sha256(key, data);
        assert!(hmac_sha256_verify(key, data, &tag));
        assert!(!hmac_sha256_verify(key, b"tampered", &tag));
    }

    #[test]
    fn test_merkle_root_empty() {
        assert_eq!(merkle_root(&[]), ZERO_HASH);
    }

    #[test]
    fn test_merkle_root_single() {
        let leaf = sha256(b"leaf");
        assert_eq!(merkle_root(&[leaf]), leaf);
    }

    #[test]
    fn test_merkle_root_two_leaves() {
        let a = sha256(b"a");
        let b = sha256(b"b");
        let root = merkle_root(&[a, b]);

        // Manual: SHA256(a || b)
        let mut hasher = Sha256::new();
        hasher.update(a);
        hasher.update(b);
        let expected: Hash256 = hasher.finalize().into();
        assert_eq!(root, expected);
    }

    #[test]
    fn test_merkle_root_odd_leaves() {
        let a = sha256(b"a");
        let b = sha256(b"b");
        let c = sha256(b"c");
        let root = merkle_root(&[a, b, c]);

        // Level 1: [H(a||b), c]
        let mut hasher = Sha256::new();
        hasher.update(a);
        hasher.update(b);
        let ab: Hash256 = hasher.finalize().into();

        // Level 2: H(ab || c)
        let mut hasher = Sha256::new();
        hasher.update(ab);
        hasher.update(c);
        let expected: Hash256 = hasher.finalize().into();
        assert_eq!(root, expected);
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
    }
}
