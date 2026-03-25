/// @file crypto.h
/// @brief Cryptographic utilities: SHA-256 hashing and HMAC-SHA256 signing.
///
/// Used to hash probe outputs for comparison against golden values and to
/// sign ProbeResult batches for tamper detection in transit to the
/// Correlation Engine.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#ifndef __CUDACC__
#include <span>
#endif
#include <string>
#include <string_view>

namespace sentinel::util {

/// SHA-256 digest size in bytes.
inline constexpr std::size_t kSha256DigestLen = 32;

/// Type alias for a 256-bit hash.
using Sha256Digest = std::array<uint8_t, kSha256DigestLen>;

/// Compute SHA-256 over a contiguous byte range.
///
/// @param data  Pointer to input data.
/// @param len   Length of input data in bytes.
/// @return      The 32-byte SHA-256 digest.
[[nodiscard]] Sha256Digest sha256(const void* data, std::size_t len);

#ifndef __CUDACC__
/// Compute SHA-256 over a span of bytes.
[[nodiscard]] Sha256Digest sha256(std::span<const uint8_t> data);
#endif

/// Incremental SHA-256 hasher for computing digests over multiple buffers
/// (e.g., streaming probe output).
class Sha256Hasher {
public:
    Sha256Hasher();
    ~Sha256Hasher();

    Sha256Hasher(const Sha256Hasher&) = delete;
    Sha256Hasher& operator=(const Sha256Hasher&) = delete;
    Sha256Hasher(Sha256Hasher&& other) noexcept;
    Sha256Hasher& operator=(Sha256Hasher&& other) noexcept;

    /// Feed data into the hash computation.
    void update(const void* data, std::size_t len);

#ifndef __CUDACC__
    /// Feed a span of bytes.
    void update(std::span<const uint8_t> data);
#endif

    /// Finalize and return the digest. The hasher is consumed; further
    /// update() calls are undefined behavior.
    [[nodiscard]] Sha256Digest finalize();

private:
    struct Impl;
    Impl* impl_;
};

/// Compute HMAC-SHA256 over data with the given key.
///
/// @param key   HMAC key bytes.
/// @param data  Message to authenticate.
/// @return      The 32-byte HMAC-SHA256 tag.
#ifndef __CUDACC__
[[nodiscard]] Sha256Digest hmac_sha256(std::span<const uint8_t> key,
                                        std::span<const uint8_t> data);

/// Convenience overload for string key and data.
[[nodiscard]] Sha256Digest hmac_sha256(std::string_view key,
                                        std::span<const uint8_t> data);
#endif

/// Constant-time comparison of two digests.
[[nodiscard]] bool digest_equal(const Sha256Digest& a, const Sha256Digest& b);

/// Convert a digest to a lowercase hex string.
[[nodiscard]] std::string digest_to_hex(const Sha256Digest& digest);

/// Parse a hex string into a digest. Returns a zero-filled digest on error.
[[nodiscard]] Sha256Digest hex_to_digest(std::string_view hex);

}  // namespace sentinel::util
