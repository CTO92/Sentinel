/// @file ring_buffer.h
/// @brief Lock-free single-producer, single-consumer ring buffer.
///
/// Designed for streaming probe results and telemetry from producer threads
/// to the reporter thread without contention. Capacity is fixed at
/// construction and must be a power of two.

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>

namespace sentinel::util {

/// Lock-free SPSC ring buffer with power-of-two capacity.
///
/// @tparam T      Element type; must be nothrow move-constructible.
/// @tparam Cap    Capacity (must be a power of two).
///
/// Memory ordering rationale:
///   - The writer publishes via release on `write_pos_`; the reader
///     acquires on `write_pos_` to see the written element.
///   - The reader publishes via release on `read_pos_`; the writer
///     acquires on `read_pos_` to reclaim the slot.
template <typename T, std::size_t Cap>
    requires(Cap > 0 && (Cap & (Cap - 1)) == 0 &&
             std::is_nothrow_move_constructible_v<T>)
class RingBuffer {
public:
    static constexpr std::size_t kCapacity = Cap;

    RingBuffer() noexcept : write_pos_{0}, read_pos_{0} {}

    ~RingBuffer() {
        // Destroy any elements still in the buffer.
        while (pop().has_value()) {
        }
    }

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;
    RingBuffer(RingBuffer&&) = delete;
    RingBuffer& operator=(RingBuffer&&) = delete;

    /// Attempt to push an element. Returns false if full.
    /// Must be called only from the single producer thread.
    [[nodiscard]] bool push(T&& item) noexcept {
        const std::size_t cur_write = write_pos_.load(std::memory_order_relaxed);
        const std::size_t next_write = (cur_write + 1) & kMask;

        // Full if the next write position equals the current read position.
        if (next_write == read_pos_.load(std::memory_order_acquire)) {
            return false;
        }

        new (&storage_[cur_write]) T(std::move(item));
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }

    /// Attempt to push by copying. Returns false if full.
    [[nodiscard]] bool push(const T& item) noexcept
        requires std::is_nothrow_copy_constructible_v<T>
    {
        const std::size_t cur_write = write_pos_.load(std::memory_order_relaxed);
        const std::size_t next_write = (cur_write + 1) & kMask;

        if (next_write == read_pos_.load(std::memory_order_acquire)) {
            return false;
        }

        new (&storage_[cur_write]) T(item);
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }

    /// Attempt to pop an element. Returns std::nullopt if empty.
    /// Must be called only from the single consumer thread.
    [[nodiscard]] std::optional<T> pop() noexcept {
        const std::size_t cur_read = read_pos_.load(std::memory_order_relaxed);

        // Empty if read position equals write position.
        if (cur_read == write_pos_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }

        T* elem = reinterpret_cast<T*>(&storage_[cur_read]);
        std::optional<T> result{std::move(*elem)};
        elem->~T();

        read_pos_.store((cur_read + 1) & kMask, std::memory_order_release);
        return result;
    }

    /// Returns the number of elements currently in the buffer.
    /// This is an estimate when called from a non-participant thread.
    [[nodiscard]] std::size_t size() const noexcept {
        const std::size_t w = write_pos_.load(std::memory_order_acquire);
        const std::size_t r = read_pos_.load(std::memory_order_acquire);
        return (w - r) & kMask;
    }

    /// Returns true if the buffer is empty (approximate).
    [[nodiscard]] bool empty() const noexcept {
        return write_pos_.load(std::memory_order_acquire) ==
               read_pos_.load(std::memory_order_acquire);
    }

    /// Returns true if the buffer is full (approximate).
    [[nodiscard]] bool full() const noexcept {
        const std::size_t w = write_pos_.load(std::memory_order_acquire);
        const std::size_t r = read_pos_.load(std::memory_order_acquire);
        return ((w + 1) & kMask) == r;
    }

    /// Returns the fixed capacity.
    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return kCapacity;
    }

private:
    static constexpr std::size_t kMask = Cap - 1;

    // Cache-line padding to prevent false sharing between producer and consumer.
    alignas(64) std::atomic<std::size_t> write_pos_;
    alignas(64) std::atomic<std::size_t> read_pos_;

    // Storage for elements. Using aligned_storage to avoid default-constructing
    // every slot.
    alignas(alignof(T)) std::aligned_storage_t<sizeof(T), alignof(T)> storage_[Cap];
};

/// Convenience alias for a ring buffer that holds ~10 seconds of probe results
/// at maximum throughput (assuming up to ~4096 results per second).
/// 8192 is the next power of two above 4096*10/5 (batched at 500ms).
template <typename T>
using ProbeRingBuffer = RingBuffer<T, 8192>;

/// Convenience alias for telemetry ring buffer (1 sample/sec, 10 sec = 16 slots).
template <typename T>
using TelemetryRingBuffer = RingBuffer<T, 16>;

}  // namespace sentinel::util
