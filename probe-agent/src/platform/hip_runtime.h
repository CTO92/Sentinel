/// @file hip_runtime.h
/// @brief HIP/ROCm runtime abstraction layer for AMD GPU support.
///
/// Provides the same RAII interface as cuda_runtime.h but targeting AMD GPUs
/// via the HIP runtime. This file is compiled only when SENTINEL_ENABLE_ROCM
/// is set in the CMake configuration. HIP is source-level compatible with CUDA
/// for most kernel code, so probe kernels can be compiled for both backends.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

/// RAII wrapper for a HIP event (timing and synchronization).
class HipEvent {
public:
    explicit HipEvent(unsigned int flags = hipEventDefault) {
        HIP_CHECK(hipEventCreateWithFlags(&event_, flags));
    }

    ~HipEvent() {
        if (event_) {
            hipEventDestroy(event_);
        }
    }

    HipEvent(const HipEvent&) = delete;
    HipEvent& operator=(const HipEvent&) = delete;

    HipEvent(HipEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    HipEvent& operator=(HipEvent&& other) noexcept {
        if (this != &other) {
            if (event_) hipEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] hipEvent_t get() const noexcept { return event_; }

    void record(hipStream_t stream = nullptr) const {
        HIP_CHECK(hipEventRecord(event_, stream));
    }

    void synchronize() const {
        HIP_CHECK(hipEventSynchronize(event_));
    }

    /// @return Elapsed time in milliseconds between start and this event.
    [[nodiscard]] float elapsed_ms(const HipEvent& start) const {
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

private:
    hipEvent_t event_ = nullptr;
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

    /// Copy data from host to device.
    void copy_from_host(const void* src, std::size_t bytes, hipStream_t stream = nullptr) {
        HIP_CHECK(hipMemcpyAsync(ptr_, src, bytes, hipMemcpyHostToDevice, stream));
    }

    /// Copy data from device to host.
    void copy_to_host(void* dst, std::size_t bytes, hipStream_t stream = nullptr) const {
        HIP_CHECK(hipMemcpyAsync(dst, ptr_, bytes, hipMemcpyDeviceToHost, stream));
    }

    /// Fill device memory with a byte pattern.
    void memset(int value, std::size_t bytes, hipStream_t stream = nullptr) {
        HIP_CHECK(hipMemsetAsync(ptr_, value, bytes, stream));
    }

private:
    void* ptr_ = nullptr;
    std::size_t size_ = 0;
};

/// RAII wrapper for HIP host-pinned memory.
class HipPinnedBuffer {
public:
    HipPinnedBuffer() = default;

    explicit HipPinnedBuffer(std::size_t bytes) : size_(bytes) {
        HIP_CHECK(hipHostMalloc(&ptr_, bytes, hipHostMallocDefault));
    }

    ~HipPinnedBuffer() {
        if (ptr_) {
            hipHostFree(ptr_);
        }
    }

    HipPinnedBuffer(const HipPinnedBuffer&) = delete;
    HipPinnedBuffer& operator=(const HipPinnedBuffer&) = delete;

    HipPinnedBuffer(HipPinnedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    HipPinnedBuffer& operator=(HipPinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) hipHostFree(ptr_);
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

/// Set the active HIP device.
inline void set_device_hip(int device_index) {
    HIP_CHECK(hipSetDevice(device_index));
}

/// Get the current HIP device index.
inline int get_device_hip() {
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    return dev;
}

/// Get the number of HIP devices.
inline int get_device_count_hip() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    return count;
}

/// Device properties for an AMD GPU.
struct HipDeviceInfo {
    int device_index = -1;
    std::string name;
    std::string gcn_arch_name;       ///< e.g. "gfx90a" (MI210), "gfx942" (MI300X).
    int compute_units = 0;           ///< Number of Compute Units (AMD's SM equivalent).
    int max_threads_per_cu = 0;
    std::size_t global_mem_bytes = 0;
    std::size_t shared_mem_per_block = 0;
    int warp_size = 0;               ///< Wavefront size (typically 64 on AMD).
    int max_clock_mhz = 0;
    int memory_clock_mhz = 0;
    int memory_bus_width = 0;
    int pcie_domain = 0;
    int pcie_bus = 0;
    int pcie_device = 0;
};

/// Query device properties for a specific HIP device.
inline HipDeviceInfo get_device_info_hip(int device_index) {
    hipDeviceProp_t props = {};
    HIP_CHECK(hipGetDeviceProperties(&props, device_index));

    HipDeviceInfo info;
    info.device_index = device_index;
    info.name = props.name;
    info.gcn_arch_name = props.gcnArchName;
    info.compute_units = props.multiProcessorCount;
    info.max_threads_per_cu = props.maxThreadsPerMultiProcessor;
    info.global_mem_bytes = props.totalGlobalMem;
    info.shared_mem_per_block = props.sharedMemPerBlock;
    info.warp_size = props.warpSize;
    info.max_clock_mhz = props.clockRate / 1000;
    info.memory_clock_mhz = props.memoryClockRate / 1000;
    info.memory_bus_width = props.memoryBusWidth;
    info.pcie_domain = props.pciDomainID;
    info.pcie_bus = props.pciBusID;
    info.pcie_device = props.pciDeviceID;
    return info;
}

/// Enumerate all HIP devices.
inline std::vector<HipDeviceInfo> enumerate_devices_hip() {
    int count = get_device_count_hip();
    std::vector<HipDeviceInfo> devices;
    devices.reserve(count);
    for (int i = 0; i < count; ++i) {
        devices.push_back(get_device_info_hip(i));
    }
    return devices;
}

}  // namespace sentinel::platform

#else  // !SENTINEL_ENABLE_ROCM

namespace sentinel::platform {

// When ROCm is not enabled, provide empty stubs so that code referencing
// HIP types compiles without #ifdef everywhere. These throw at runtime
// if actually called.

inline void set_device_hip(int /*device_index*/) {
    throw std::runtime_error("HIP/ROCm support not compiled in");
}

inline int get_device_hip() {
    throw std::runtime_error("HIP/ROCm support not compiled in");
}

inline int get_device_count_hip() {
    return 0;  // No devices available when ROCm is not compiled in.
}

}  // namespace sentinel::platform

#endif  // SENTINEL_ENABLE_ROCM
