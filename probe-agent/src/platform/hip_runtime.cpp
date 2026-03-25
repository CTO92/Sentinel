/// @file hip_runtime.cpp
/// @brief HIP/ROCm runtime abstraction — compilation unit.
///
/// This file ensures the translation unit exists for the build system.
/// All HIP functionality is header-only or conditionally compiled via
/// SENTINEL_ENABLE_ROCM.

#include "platform/hip_runtime.h"
