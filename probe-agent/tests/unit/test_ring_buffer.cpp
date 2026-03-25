/// @file test_ring_buffer.cpp
/// @brief Unit tests for the lock-free SPSC ring buffer.

#include "util/ring_buffer.h"

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace sentinel::util {
namespace {

TEST(RingBufferTest, BasicPushPop) {
    RingBuffer<int, 8> buffer;

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0u);

    EXPECT_TRUE(buffer.push(42));
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size(), 1u);

    auto val = buffer.pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
    EXPECT_TRUE(buffer.empty());
}

TEST(RingBufferTest, FillToCapacity) {
    RingBuffer<int, 4> buffer;  // 4 slots, usable capacity is 3 (one sentinel).

    EXPECT_TRUE(buffer.push(1));
    EXPECT_TRUE(buffer.push(2));
    EXPECT_TRUE(buffer.push(3));
    EXPECT_FALSE(buffer.push(4));  // Full (capacity - 1 = 3 items).
    EXPECT_TRUE(buffer.full());
}

TEST(RingBufferTest, WrapAround) {
    RingBuffer<int, 4> buffer;

    // Fill 3 slots.
    EXPECT_TRUE(buffer.push(1));
    EXPECT_TRUE(buffer.push(2));
    EXPECT_TRUE(buffer.push(3));

    // Pop 2.
    EXPECT_EQ(*buffer.pop(), 1);
    EXPECT_EQ(*buffer.pop(), 2);

    // Push 2 more (wrap around).
    EXPECT_TRUE(buffer.push(4));
    EXPECT_TRUE(buffer.push(5));

    // Pop remaining 3.
    EXPECT_EQ(*buffer.pop(), 3);
    EXPECT_EQ(*buffer.pop(), 4);
    EXPECT_EQ(*buffer.pop(), 5);

    EXPECT_TRUE(buffer.empty());
}

TEST(RingBufferTest, EmptyPopReturnsNullopt) {
    RingBuffer<int, 4> buffer;
    auto val = buffer.pop();
    EXPECT_FALSE(val.has_value());
}

TEST(RingBufferTest, MoveOnlyTypes) {
    RingBuffer<std::unique_ptr<int>, 4> buffer;

    auto p = std::make_unique<int>(99);
    EXPECT_TRUE(buffer.push(std::move(p)));

    auto val = buffer.pop();
    ASSERT_TRUE(val.has_value());
    ASSERT_NE(*val, nullptr);
    EXPECT_EQ(**val, 99);
}

TEST(RingBufferTest, StringType) {
    RingBuffer<std::string, 8> buffer;

    EXPECT_TRUE(buffer.push(std::string("hello")));
    EXPECT_TRUE(buffer.push(std::string("world")));

    auto v1 = buffer.pop();
    ASSERT_TRUE(v1.has_value());
    EXPECT_EQ(*v1, "hello");

    auto v2 = buffer.pop();
    ASSERT_TRUE(v2.has_value());
    EXPECT_EQ(*v2, "world");
}

TEST(RingBufferTest, Capacity) {
    RingBuffer<int, 16> buffer;
    EXPECT_EQ(buffer.capacity(), 16u);
}

TEST(RingBufferTest, ConcurrentProducerConsumer) {
    // Stress test: single producer, single consumer, 1M items.
    constexpr int kNumItems = 1'000'000;
    RingBuffer<int, 1024> buffer;

    std::atomic<bool> producer_done{false};
    std::atomic<int> consumed_count{0};
    int64_t sum_produced = 0;
    std::atomic<int64_t> sum_consumed{0};

    // Producer thread.
    std::thread producer([&]() {
        for (int i = 0; i < kNumItems; ++i) {
            while (!buffer.push(i)) {
                // Spin until slot available.
                std::this_thread::yield();
            }
            sum_produced += i;
        }
        producer_done.store(true, std::memory_order_release);
    });

    // Consumer thread.
    std::thread consumer([&]() {
        while (true) {
            auto val = buffer.pop();
            if (val.has_value()) {
                sum_consumed.fetch_add(*val, std::memory_order_relaxed);
                consumed_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                if (producer_done.load(std::memory_order_acquire) && buffer.empty()) {
                    break;
                }
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    EXPECT_EQ(consumed_count.load(), kNumItems);
    EXPECT_EQ(sum_consumed.load(), sum_produced);
}

TEST(RingBufferTest, DestructorCleanup) {
    // Verify that elements still in the buffer are properly destroyed.
    static int dtor_count = 0;

    struct Counted {
        Counted() = default;
        Counted(Counted&&) noexcept = default;
        Counted& operator=(Counted&&) noexcept = default;
        ~Counted() { ++dtor_count; }
    };

    dtor_count = 0;
    {
        RingBuffer<Counted, 8> buffer;
        buffer.push(Counted{});
        buffer.push(Counted{});
        buffer.push(Counted{});
        // Pop one.
        buffer.pop();
        // 2 elements remain when buffer is destroyed.
    }
    // The 2 remaining elements + 1 popped + temporaries should all be destroyed.
    EXPECT_GE(dtor_count, 3);
}

}  // namespace
}  // namespace sentinel::util
