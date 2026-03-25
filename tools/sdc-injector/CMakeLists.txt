cmake_minimum_required(VERSION 3.24)
project(sdc_injector LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Require CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Detect GPU architectures — default to Ampere + Hopper
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

# --------------------------------------------------------------------------- #
#  SDC Injector shared library
# --------------------------------------------------------------------------- #
add_library(sdc_injector SHARED
    src/injector.cu
)

target_include_directories(sdc_injector PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(sdc_injector PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

# Enable relocatable device code for separate compilation
set_target_properties(sdc_injector PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "sdc_injector"
    VERSION 1.0.0
    SOVERSION 1
)

# --------------------------------------------------------------------------- #
#  CLI executable (thin wrapper around the library)
# --------------------------------------------------------------------------- #
add_executable(sdc_injector_cli
    src/injector.cu
)

target_compile_definitions(sdc_injector_cli PRIVATE SDC_INJECTOR_BUILD_CLI=1)

target_link_libraries(sdc_injector_cli PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(sdc_injector_cli PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "sdc-injector"
)

# --------------------------------------------------------------------------- #
#  Install rules
# --------------------------------------------------------------------------- #
install(TARGETS sdc_injector sdc_injector_cli
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib
    RUNTIME  DESTINATION bin
)

install(FILES src/injector.cu DESTINATION include/sdc_injector RENAME injector.cuh)
