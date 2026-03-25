cmake_minimum_required(VERSION 3.22)
project(sentinel-probe-agent
    VERSION   0.1.0
    LANGUAGES CXX CUDA
)

# ── C++20 / CUDA standards ────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set(CMAKE_CUDA_STANDARD         17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Volta (70), Turing (75), Ampere (80, 86), Ada Lovelace (89), Hopper (90)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)

# ── Build options ─────────────────────────────────────────────────────
option(SENTINEL_BUILD_TESTS   "Build unit and integration tests"  ON)
option(SENTINEL_ENABLE_ROCM   "Build with ROCm/HIP support"      OFF)
option(SENTINEL_SANITIZERS    "Enable ASan/UBSan (debug builds)"  OFF)

# ── Conan-generated paths (if using Conan) ────────────────────────────
if(EXISTS "${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
    include("${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
endif()

# ── Find required packages ────────────────────────────────────────────
find_package(CUDAToolkit 12.0 REQUIRED)

# gRPC & Protobuf (optional – only needed for the communication layer)
find_package(protobuf CONFIG QUIET)
find_package(gRPC     CONFIG QUIET)
find_package(Threads  REQUIRED)

# Logging
find_package(spdlog CONFIG QUIET)

# Metrics
find_package(prometheus-cpp CONFIG QUIET)

# Cryptography
find_package(OpenSSL QUIET)

# JSON (for config / golden answers)
find_package(nlohmann_json CONFIG QUIET)

# ── Fallback: if key dependencies not found, provide stub targets ──────
if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found — providing stub logging target")
    add_library(spdlog::spdlog INTERFACE IMPORTED)
endif()
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found — providing stub target")
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
endif()
if(NOT OpenSSL_FOUND)
    message(STATUS "OpenSSL not found — crypto functions will be stubbed")
    add_library(OpenSSL::SSL INTERFACE IMPORTED)
    add_library(OpenSSL::Crypto INTERFACE IMPORTED)
endif()
if(NOT TARGET prometheus-cpp::core)
    message(STATUS "prometheus-cpp not found — metrics will be disabled")
    add_library(prometheus-cpp::core INTERFACE IMPORTED)
    add_library(prometheus-cpp::pull INTERFACE IMPORTED)
endif()

# ── Protobuf / gRPC code generation (only when both are found) ──────
if(protobuf_FOUND AND gRPC_FOUND)
    set(PROTO_DIR "${CMAKE_SOURCE_DIR}/../proto")
    set(PROTO_FILES
        "${PROTO_DIR}/sentinel/v1/common.proto"
        "${PROTO_DIR}/sentinel/v1/probe.proto"
        "${PROTO_DIR}/sentinel/v1/telemetry.proto"
        "${PROTO_DIR}/sentinel/v1/health.proto"
    )

    set(PROTO_OUT_DIR "${CMAKE_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${PROTO_OUT_DIR}")

    # Locate protoc and grpc_cpp_plugin
    find_program(_PROTOC protoc REQUIRED)
    find_program(_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

    set(PROTO_SRCS "")
    set(PROTO_HDRS "")
    set(GRPC_SRCS  "")
    set(GRPC_HDRS  "")

    foreach(_proto ${PROTO_FILES})
        get_filename_component(_proto_name "${_proto}" NAME_WLE)
        get_filename_component(_proto_path "${_proto}" DIRECTORY)

        set(_pb_cc   "${PROTO_OUT_DIR}/sentinel/v1/${_proto_name}.pb.cc")
        set(_pb_h    "${PROTO_OUT_DIR}/sentinel/v1/${_proto_name}.pb.h")
        set(_grpc_cc "${PROTO_OUT_DIR}/sentinel/v1/${_proto_name}.grpc.pb.cc")
        set(_grpc_h  "${PROTO_OUT_DIR}/sentinel/v1/${_proto_name}.grpc.pb.h")

        add_custom_command(
            OUTPUT "${_pb_cc}" "${_pb_h}" "${_grpc_cc}" "${_grpc_h}"
            COMMAND "${_PROTOC}"
                --proto_path="${PROTO_DIR}"
                --cpp_out="${PROTO_OUT_DIR}"
                --grpc_out="${PROTO_OUT_DIR}"
                --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN}"
                "${_proto}"
            DEPENDS "${_proto}"
            COMMENT "Generating protobuf/gRPC sources for ${_proto_name}"
        )
        list(APPEND PROTO_SRCS "${_pb_cc}")
        list(APPEND PROTO_HDRS "${_pb_h}")
        list(APPEND GRPC_SRCS  "${_grpc_cc}")
        list(APPEND GRPC_HDRS  "${_grpc_h}")
    endforeach()

    add_library(sentinel_proto STATIC ${PROTO_SRCS} ${GRPC_SRCS})
    target_include_directories(sentinel_proto PUBLIC "${PROTO_OUT_DIR}")
    target_link_libraries(sentinel_proto
        PUBLIC
            protobuf::libprotobuf
            gRPC::grpc++
    )
else()
    message(STATUS "protobuf or gRPC not found — building without gRPC communication layer")
endif()

# ── Agent library (non-CUDA sources) ─────────────────────────────────
set(AGENT_SOURCES
    src/agent/agent.cpp
    src/agent/scheduler.cpp
    src/agent/config_manager.cpp
    src/telemetry/nvml_collector.cpp
    src/telemetry/rocm_smi_collector.cpp
    src/telemetry/thermal_monitor.cpp
    src/telemetry/voltage_monitor.cpp
    src/platform/cuda_runtime.cpp
    src/platform/hip_runtime.cpp
    src/platform/platform_detect.cpp
    src/util/logging.cpp
    src/util/crypto.cpp
)

# Prometheus metrics: use real impl when available, no-op fallback otherwise
if(prometheus-cpp_FOUND)
    list(APPEND AGENT_SOURCES src/util/metrics.cpp)
else()
    list(APPEND AGENT_SOURCES src/util/metrics_noop.cpp)
endif()

# gRPC-dependent sources are only compiled when protobuf/gRPC are available
if(protobuf_FOUND AND gRPC_FOUND)
    list(APPEND AGENT_SOURCES src/agent/grpc_client.cpp)
endif()

add_library(sentinel_agent_lib STATIC ${AGENT_SOURCES})
target_include_directories(sentinel_agent_lib
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(sentinel_agent_lib
    PUBLIC
        spdlog::spdlog
        prometheus-cpp::core
        prometheus-cpp::pull
        OpenSSL::SSL
        OpenSSL::Crypto
        nlohmann_json::nlohmann_json
        CUDA::cudart
        CUDA::nvml
        Threads::Threads
)
if(prometheus-cpp_FOUND)
    target_compile_definitions(sentinel_agent_lib PUBLIC SENTINEL_HAS_PROMETHEUS=1)
endif()

if(protobuf_FOUND AND gRPC_FOUND)
    target_include_directories(sentinel_agent_lib PUBLIC "${PROTO_OUT_DIR}")
    target_link_libraries(sentinel_agent_lib PUBLIC sentinel_proto)
    target_compile_definitions(sentinel_agent_lib PUBLIC SENTINEL_HAS_GRPC=1)
endif()

# ── CUDA probe library ───────────────────────────────────────────────
set(CUDA_PROBE_SOURCES
    src/probes/fma_probe.cu
    src/probes/tensor_core_probe.cu
    src/probes/transcendental_probe.cu
    src/probes/aes_probe.cu
    src/probes/memory_probe.cu
    src/probes/register_file_probe.cu
    src/probes/shared_memory_probe.cu
    src/probes/sm_affinity.cu
)

add_library(sentinel_probes STATIC ${CUDA_PROBE_SOURCES})
target_include_directories(sentinel_probes
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
set_target_properties(sentinel_probes PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(sentinel_probes
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        OpenSSL::Crypto
)
target_compile_options(sentinel_probes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        -Xcompiler=-Wall,-Wextra
    >
)

# ── Main executable ──────────────────────────────────────────────────
add_executable(sentinel-probe-agent src/main.cpp)
target_link_libraries(sentinel-probe-agent
    PRIVATE
        sentinel_agent_lib
        sentinel_probes
)

# ── Tests ─────────────────────────────────────────────────────────────
option(SENTINEL_BUILD_TESTS "Build test targets" ON)
if(SENTINEL_BUILD_TESTS OR BUILD_TESTING)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        enable_testing()
        include(GoogleTest)

        # Unit tests (C++ only)
        set(UNIT_TESTS
            tests/unit/test_scheduler.cpp
            tests/unit/test_ring_buffer.cpp
        )
        add_executable(sentinel_unit_tests ${UNIT_TESTS})
        set(_UNIT_TEST_LIBS sentinel_agent_lib GTest::gtest GTest::gtest_main)
        if(TARGET GTest::gmock)
            list(APPEND _UNIT_TEST_LIBS GTest::gmock)
        endif()
        target_link_libraries(sentinel_unit_tests PRIVATE ${_UNIT_TEST_LIBS})
        gtest_discover_tests(sentinel_unit_tests)

        # CUDA unit tests
        set(CUDA_UNIT_TESTS
            tests/unit/test_fma_probe.cpp
            tests/unit/test_sm_affinity.cpp
        )
        add_executable(sentinel_cuda_unit_tests ${CUDA_UNIT_TESTS})
        target_link_libraries(sentinel_cuda_unit_tests
            PRIVATE
                sentinel_agent_lib
                sentinel_probes
                GTest::gtest
                GTest::gtest_main
        )
        gtest_discover_tests(sentinel_cuda_unit_tests)

        # Integration tests (gRPC tests only when protobuf/gRPC are available)
        set(INTEGRATION_TESTS
            tests/integration/test_full_probe_cycle.cpp
        )
        if(protobuf_FOUND AND gRPC_FOUND)
            list(APPEND INTEGRATION_TESTS tests/integration/test_agent_grpc.cpp)
        endif()
        add_executable(sentinel_integration_tests ${INTEGRATION_TESTS})
        target_link_libraries(sentinel_integration_tests
            PRIVATE
                sentinel_agent_lib
                sentinel_probes
                GTest::gtest
                GTest::gtest_main
        )
        gtest_discover_tests(sentinel_integration_tests)
    else()
        message(STATUS "GTest not found — skipping test targets")
    endif()
endif()

# ── Install rules ────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS sentinel-probe-agent
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(FILES
    "${CMAKE_SOURCE_DIR}/packaging/sentinel-probe-agent.service"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/systemd/system"
)
