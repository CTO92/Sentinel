/// @file cuda_runtime.h
/// @brief CUDA runtime abstraction layer.
///
/// Provides RAII wrappers for CUDA resources (streams, events, memory),
/// error checking macros, and GPU context management. Ensures deterministic
/// teardown order and consistent error handling.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace sentinel::platform {

/// Exception thrown on unrecoverable CUDA errors.
class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t code, const char* file, int line)
        : std::runtime_error(format_message(code, file, line)),
          code_(code) {}

    [[nodiscard]] cudaError_t code() const noexcept { return code_; }

private:
    static std::string format_message(cudaError_t code, const char* file, int line) {
        return std::string("CUDA error ") + cudaGetErrorName(code) +
               " (" + cudaGetErrorString(code) + ") at " + file + ":" +
               std::to_string(line);
    }

    cudaError_t code_;
};

/// Check a CUDA API call and throw CudaError on failure.
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _err = (call);                                        \
        if (_err != cudaSuccess) {                                        \
            throw ::sentinel::platform::CudaError(_err, __FILE__, __LINE__); \
        }                                                                 \
    } while (0)

/// Check a CUDA API call, log and return false on failure (non-throwing).
#define CUDA_CHECK_NOTHROW(call, logger)                                  \
    [&]() -> bool {                                                       \
        cudaError_t _err = (call);                                        \
        if (_err != cudaSuccess) {                                        \
            SPDLOG_LOGGER_ERROR(logger, "CUDA error {} ({}) at {}:{}",    \
                                cudaGetErrorName(_err),                   \
                                cudaGetErrorString(_err),                 \
                                __FILE__, __LINE__);                      \
            return false;                                                 \
        }                                                                 \
        return true;                                                      \
    }()

/// RAII wrapper for a CUDA stream.
class CudaStream {
public:
    /// Create a new CUDA stream on the current device.
    /// @param flags  cudaStreamDefault or cudaStreamNonBlocking.
    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~CudaStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    [[nodiscard]] operator cudaStream_t() const noexcept { return stream_; }

    /// Synchronize this stream.
    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

    /// Query whether all operations on this stream have completed.
    [[nodiscard]] bool is_complete() const {
        cudaError_t err = cudaStreamQuery(stream_);
        if (err == cudaSuccess) return true;
        if (err == cudaErrorNotReady) return false;
        CUDA_CHECK(err);  // Unexpected error.
        return false;
    }

private:
    cudaStream_t stream_ = nullptr;
};

/// RAII wrapper for a CUDA event.
class CudaEvent {
public:
    explicit CudaEvent(unsigned int flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    ~CudaEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

    /// Record this event on a stream.
    void record(cudaStream_t stream) { CUDA_CHECK(cudaEventRecord(event_, stream)); }

    /// Synchronize on this event.
    void synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }

    /// Elapsed time in milliseconds between two events.
    [[nodiscard]] static float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, stop.event_));
        return ms;
    }

private:
    cudaEvent_t event_ = nullptr;
};

/// RAII wrapper for device memory.
class CudaDeviceBuffer {
public:
    CudaDeviceBuffer() = default;

    explicit CudaDeviceBuffer(std::size_t bytes) : size_(bytes) {
        CUDA_CHECK(cudaMalloc(&ptr_, bytes));
    }

    ~CudaDeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    [[nodiscard]] void* get() const noexcept { return ptr_; }

    template <typename T>
    [[nodiscard]] T* as() const noexcept { return static_cast<T*>(ptr_); }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// Copy from host to this device buffer.
    void copy_from_host(const void* src, std::size_t bytes, cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, src, bytes, cudaMemcpyHostToDevice, stream));
    }

    /// Copy from this device buffer to host.
    void copy_to_host(void* dst, std::size_t bytes, cudaStream_t stream = nullptr) const {
        CUDA_CHECK(cudaMemcpyAsync(dst, ptr_, bytes, cudaMemcpyDeviceToHost, stream));
    }

private:
    void* ptr_ = nullptr;
    std::size_t size_ = 0;
};

/// RAII wrapper for pinned (page-locked) host memory.
class CudaPinnedBuffer {
public:
    CudaPinnedBuffer() = default;

    explicit CudaPinnedBuffer(std::size_t bytes) : size_(bytes) {
        CUDA_CHECK(cudaMallocHost(&ptr_, bytes));
    }

    ~CudaPinnedBuffer() {
        if (ptr_) {
            cudaFreeHost(ptr_);
        }
    }

    CudaPinnedBuffer(const CudaPinnedBuffer&) = delete;
    CudaPinnedBuffer& operator=(const CudaPinnedBuffer&) = delete;

    CudaPinnedBuffer(CudaPinnedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaPinnedBuffer& operator=(CudaPinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    [[nodiscard]] void* get() const noexcept { return ptr_; }

    template <typename T>
    [[nodiscard]] T* as() const noexcept { return static_cast<T*>(ptr_); }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    void* ptr_ = nullptr;
    std::size_t size_ = 0;
};

/// Set the current CUDA device. Thread-safe (CUDA manages per-thread state).
inline void set_device(int device_index) {
    CUDA_CHECK(cudaSetDevice(device_index));
}

/// Get the current CUDA device index.
[[nodiscard]] inline int get_device() {
    int dev = -1;
    CUDA_CHECK(cudaGetDevice(&dev));
    return dev;
}

/// Reset the current CUDA device (destroys all allocations and state).
inline void reset_device() {
    CUDA_CHECK(cudaDeviceReset());
}

}  // namespace sentinel::platform
