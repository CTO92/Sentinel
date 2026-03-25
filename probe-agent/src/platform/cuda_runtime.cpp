/// @file cuda_runtime.cpp
/// @brief CUDA runtime abstraction — compilation unit for non-inline symbols.
///
/// Most functionality is header-only (RAII wrappers), but this file
/// ensures the translation unit exists for the build system and provides
/// any implementation that cannot be inlined.

#include "platform/cuda_runtime.h"
