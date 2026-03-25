/// @file hip_runtime.h
/// @brief HIP/ROCm runtime abstraction layer (stub for future ROCm support).
///
/// Provides the same interface as cuda_runtime.h but targeting AMD GPUs
/// via the HIP runtime. This file is compiled only when SENTINEL_ENABLE_ROCM
/// is set in the CMake configuration.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef SENTINEL_ENABLE_ROCM

#include <hip/hip_runtime.h>

namespace sentinel::platform {

/// Exception thrown on unrecoverable HIP errors.
class HipError : public std::runtime_error {
public:
    HipError(hipError_t code, const char* file, int line)
        : std::runtime_error(format_message(code, file, line)),
          code_(code) {}

    [[nodiscard]] hipError_t code() const noexcept { return code_; }

private:
    static std::string format_message(hipError_t code, const char* file, int line) {
        return std::string("HIP error ") + hipGetErrorName(code) +
               " (" + hipGetErrorString(code) + ") at " + file + ":" +
               std::to_string(line);
    }

    hipError_t code_;
};

#define HIP_CHECK(call)                                                   \
    do {                                                                  \
        hipError_t _err = (call);                                         \
        if (_err != hipSuccess) {                                         \
            throw ::sentinel::platform::HipError(_err, __FILE__, __LINE__); \
        }                                                                 \
    } while (0)

/// RAII wrapper for a HIP stream.
class HipStream {
public:
    explicit HipStream(unsigned int flags = hipStreamNonBlocking) {
        HIP_CHECK(hipStreamCreateWithFlags(&stream_, flags));
    }

    ~HipStream() {
        if (stream_) {
            hipStreamDestroy(stream_);
        }
    }

    HipStream(const HipStream&) = delete;
    HipStream& operator=(const HipStream&) = delete;

    HipStream(HipStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    HipStream& operator=(HipStream&& other) noexcept {
        if (this != &other) {
            if (stream_) hipStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] hipStream_t get() const noexcept { return stream_; }
    void synchronize() const { HIP_CHECK(hipStreamSynchronize(stream_)); }

private:
    hipStream_t stream_ = nullptr;
};

/// RAII wrapper for HIP device memory.
class HipDeviceBuffer {
public:
    HipDeviceBuffer() = default;

    explicit HipDeviceBuffer(std::size_t bytes) : size_(bytes) {
        HIP_CHECK(hipMalloc(&ptr_, bytes));
    }

    ~HipDeviceBuffer() {
        if (ptr_) {
            hipFree(ptr_);
        }
    }

    HipDeviceBuffer(const HipDeviceBuffer&) = delete;
    HipDeviceBuffer& operator=(const HipDeviceBuffer&) = delete;

    HipDeviceBuffer(HipDeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    HipDeviceBuffer& operator=(HipDeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) hipFree(ptr_);
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

inline void set_device_hip(int device_index) {
    HIP_CHECK(hipSetDevice(device_index));
}

}  // namespace sentinel::platform

#else  // !SENTINEL_ENABLE_ROCM

namespace sentinel::platform {

// When ROCm is not enabled, provide empty stubs so that code referencing
// HIP types compiles without #ifdef everywhere. These should never be
// called at runtime.

inline void set_device_hip(int /*device_index*/) {
    throw std::runtime_error("HIP/ROCm support not compiled in");
}

}  // namespace sentinel::platform

#endif  // SENTINEL_ENABLE_ROCM
