/// @file crypto.cpp
/// @brief SHA-256 and HMAC-SHA256 implementations backed by OpenSSL.

#include "util/crypto.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/crypto.h>

namespace sentinel::util {

// ── SHA-256 one-shot ──────────────────────────────────────────────────

Sha256Digest sha256(const void* data, std::size_t len) {
    Sha256Digest digest{};
    unsigned int out_len = 0;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        throw std::runtime_error("EVP_MD_CTX_new failed");
    }

    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data, len) != 1 ||
        EVP_DigestFinal_ex(ctx, digest.data(), &out_len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("SHA-256 computation failed");
    }

    EVP_MD_CTX_free(ctx);
    return digest;
}

Sha256Digest sha256(std::span<const uint8_t> data) {
    return sha256(data.data(), data.size());
}

// ── Incremental SHA-256 ──────────────────────────────────────────────

struct Sha256Hasher::Impl {
    EVP_MD_CTX* ctx = nullptr;

    Impl() : ctx(EVP_MD_CTX_new()) {
        if (!ctx) {
            throw std::runtime_error("EVP_MD_CTX_new failed");
        }
        if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1) {
            EVP_MD_CTX_free(ctx);
            throw std::runtime_error("EVP_DigestInit_ex failed");
        }
    }

    ~Impl() {
        if (ctx) {
            EVP_MD_CTX_free(ctx);
        }
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
};

Sha256Hasher::Sha256Hasher() : impl_(new Impl()) {}

Sha256Hasher::~Sha256Hasher() {
    delete impl_;
}

Sha256Hasher::Sha256Hasher(Sha256Hasher&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

Sha256Hasher& Sha256Hasher::operator=(Sha256Hasher&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

void Sha256Hasher::update(const void* data, std::size_t len) {
    if (EVP_DigestUpdate(impl_->ctx, data, len) != 1) {
        throw std::runtime_error("EVP_DigestUpdate failed");
    }
}

void Sha256Hasher::update(std::span<const uint8_t> data) {
    update(data.data(), data.size());
}

Sha256Digest Sha256Hasher::finalize() {
    Sha256Digest digest{};
    unsigned int out_len = 0;
    if (EVP_DigestFinal_ex(impl_->ctx, digest.data(), &out_len) != 1) {
        throw std::runtime_error("EVP_DigestFinal_ex failed");
    }
    return digest;
}

// ── HMAC-SHA256 ──────────────────────────────────────────────────────

Sha256Digest hmac_sha256(std::span<const uint8_t> key,
                          std::span<const uint8_t> data) {
    Sha256Digest tag{};
    unsigned int out_len = 0;

    // Use the EVP_MAC API (OpenSSL 3.x compatible).
    EVP_MAC* mac = EVP_MAC_fetch(nullptr, "HMAC", nullptr);
    if (!mac) {
        throw std::runtime_error("EVP_MAC_fetch(HMAC) failed");
    }

    EVP_MAC_CTX* ctx = EVP_MAC_CTX_new(mac);
    if (!ctx) {
        EVP_MAC_free(mac);
        throw std::runtime_error("EVP_MAC_CTX_new failed");
    }

    OSSL_PARAM params[] = {
        OSSL_PARAM_construct_utf8_string("digest",
                                          const_cast<char*>("SHA256"), 0),
        OSSL_PARAM_construct_end(),
    };

    if (EVP_MAC_init(ctx, key.data(), key.size(), params) != 1 ||
        EVP_MAC_update(ctx, data.data(), data.size()) != 1) {
        EVP_MAC_CTX_free(ctx);
        EVP_MAC_free(mac);
        throw std::runtime_error("HMAC-SHA256 init/update failed");
    }

    std::size_t actual_len = kSha256DigestLen;
    if (EVP_MAC_final(ctx, tag.data(), &actual_len, kSha256DigestLen) != 1) {
        EVP_MAC_CTX_free(ctx);
        EVP_MAC_free(mac);
        throw std::runtime_error("HMAC-SHA256 final failed");
    }

    EVP_MAC_CTX_free(ctx);
    EVP_MAC_free(mac);
    return tag;
}

Sha256Digest hmac_sha256(std::string_view key,
                          std::span<const uint8_t> data) {
    return hmac_sha256(
        std::span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(key.data()), key.size()),
        data);
}

// ── Utilities ─────────────────────────────────────────────────────────

bool digest_equal(const Sha256Digest& a, const Sha256Digest& b) {
    // CRYPTO_memcmp is constant-time to prevent timing side-channels.
    return CRYPTO_memcmp(a.data(), b.data(), kSha256DigestLen) == 0;
}

std::string digest_to_hex(const Sha256Digest& digest) {
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string result;
    result.reserve(kSha256DigestLen * 2);
    for (uint8_t byte : digest) {
        result.push_back(hex_chars[(byte >> 4) & 0x0F]);
        result.push_back(hex_chars[byte & 0x0F]);
    }
    return result;
}

Sha256Digest hex_to_digest(std::string_view hex) {
    Sha256Digest digest{};
    if (hex.size() != kSha256DigestLen * 2) {
        return digest;  // Return zeroed digest on invalid input.
    }

    auto hex_val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };

    for (std::size_t i = 0; i < kSha256DigestLen; ++i) {
        int hi = hex_val(hex[i * 2]);
        int lo = hex_val(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) {
            digest.fill(0);
            return digest;
        }
        digest[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return digest;
}

}  // namespace sentinel::util
