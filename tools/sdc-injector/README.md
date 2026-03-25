# SENTINEL SDC Injector

Controlled Silent Data Corruption (SDC) injection toolkit for validating the SENTINEL detection pipeline end-to-end.

## Overview

The SDC Injector provides deterministic fault injection into GPU memory and execution units. It allows operators and CI systems to verify that every layer of the SENTINEL framework (probe agent, inference monitor, training monitor, correlation engine) correctly detects each class of SDC.

**Safety**: All injection functions are gated behind an explicit `--enable-injection` flag. Without it, every call is a no-op. This prevents accidental corruption of production workloads.

## Components

| File | Purpose |
|------|---------|
| `src/injector.cu` | CUDA kernels for bit-flip, stuck-at, noise, register, shared-memory, and tensor-core fault injection |
| `src/scenarios.py` | Predefined test scenarios with expected detection layers and time budgets |
| `src/harness.py` | Test orchestration: runs scenarios, polls SENTINEL API, generates reports |

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;90"
cmake --build . --parallel
```

The build produces:
- `libsdc_injector.so` — shared library for programmatic use
- `sdc-injector` — CLI binary for manual testing

## Quick Start

### Self-test (verifies all injection types on local GPU)

```bash
./build/sdc-injector --enable-injection selftest
```

### Run a specific injection

```bash
./build/sdc-injector --enable-injection bitflip --offset 1024 --bit 5
./build/sdc-injector --enable-injection noise --count 4096 --sigma 0.01
./build/sdc-injector --enable-injection stuck-at --sm 7 --value 42.0 --count 4096
```

### Run the full test harness

```bash
python src/harness.py \
    --sentinel-api http://localhost:8080 \
    --scenarios all \
    --report results.json
```

### Run specific scenarios

```bash
python src/harness.py \
    --sentinel-api http://localhost:8080 \
    --scenarios single_weight_bitflip,fma_stuck_at
```

### List available scenarios

```bash
python src/harness.py --sentinel-api http://localhost:8080 --list
```

## Injection Types

| Type | Kernel | Description |
|------|--------|-------------|
| `bitflip` | `inject_bitflip` | Flip a single bit in GPU global memory |
| `stuck_at` | `inject_stuck_at` | Force FMA output to a constant on one SM |
| `noise` | `inject_noise` | Add Gaussian noise to a memory region |
| `memory_stuck_bit` | `inject_memory_stuck_bit` | Simulate a physically stuck HBM bit |
| `tensor_core` | `inject_tensor_core_corruption` | Corrupt HMMA (FP16) output |
| `register` | `inject_register_corruption` | XOR a mask into FP registers via PTX |
| `shared_memory` | `inject_shared_memory_corruption` | Corrupt a word in shared memory |

## Predefined Scenarios

| Scenario | Injection | Expected Detector | Time Budget |
|----------|-----------|-------------------|-------------|
| `single_weight_bitflip` | 1-bit flip in weight tensor | Probe Agent | 30s |
| `fma_stuck_at` | Stuck FMA on one SM | Probe Agent | 15s |
| `gradual_degradation` | Ramping noise on one GPU | Inference Monitor | 120s |
| `correlated_failure` | Multi-GPU noise | Correlation Engine | 60s |
| `byzantine_fault` | Subtle register corruption | Inference Monitor | 45s |
| `tensor_core_corruption` | HMMA bit-flip | Probe Agent | 20s |
| `memory_stuck_bit` | Stuck HBM bit | Probe Agent | 30s |

## Structured Logging

Every injection is logged to stdout in JSON:

```json
{
  "timestamp": "2026-03-24T12:00:00Z",
  "component": "sdc-injector",
  "event": "inject",
  "target": "bitflip",
  "details": {"address": "0x7f1234000", "byte_offset": 1024, "bit_position": 5}
}
```

## Library API (C)

```c
void sdc_injector_enable(bool enable);
int  inject_bitflip(void* addr, size_t offset, int bit);
int  inject_stuck_at(float* out, int count, int sm, float val);
int  inject_noise(float* addr, int count, float sigma);
int  inject_memory_stuck_bit(void* addr, size_t offset, int bit, int val);
int  inject_tensor_core_corruption(void* out, int count, int bit);
int  inject_register_corruption(float* out, int count, uint32_t mask);
int  inject_shared_memory_corruption(float* out, int count, int word, float val);
```

All functions return 0 on success, -1 on error or if injection is not enabled.

## Python API

```python
from scenarios import get_scenario, ALL_SCENARIOS
from harness import TestHarness

harness = TestHarness(sentinel_api="http://localhost:8080")
result = harness.run_scenario(get_scenario("single_weight_bitflip"))
print(result.passed)  # True if SENTINEL detected the fault
```
