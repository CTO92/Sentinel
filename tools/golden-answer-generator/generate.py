"""
SENTINEL Golden Answer Generator

Generates precomputed reference values for probe agent validation using
arbitrary-precision arithmetic (mpmath with 128-bit precision). Golden answers
are stored per GPU architecture family and used by the probe agent to verify
GPU computation correctness.

Supported test vector categories:
  - FMA (fused multiply-add): 1024 vectors spanning normal, denormal,
    exponent boundary, and mantissa-pattern ranges
  - Tensor Core: 16x16 matrix multiply test cases
  - Transcendental: 256 values per function (sin, cos, exp, log, rsqrt)
  - AES: 4KB plaintext + key + expected ciphertext
  - Memory: Data patterns for March C- test

Usage:
    python generate.py --arch ampere --output golden_ampere.json
    python generate.py --arch hopper --output golden_hopper.json
    python generate.py --all --output-dir golden/

Copyright 2025-2026 SENTINEL Authors — Apache 2.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mpmath

# 128-bit (quad) precision — ~34 decimal digits
mpmath.mp.prec = 128

ARCHITECTURES = ["ampere", "hopper", "ada", "blackwell"]


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def _float_to_hex(f: float) -> str:
    """IEEE 754 double to hex string."""
    return struct.pack("!d", f).hex()


def _float32_to_hex(f: float) -> str:
    """IEEE 754 single to hex string."""
    return struct.pack("!f", f).hex()


def _mpf_to_float64(v: mpmath.mpf) -> float:
    return float(v)


def _mpf_to_float32_hex(v: mpmath.mpf) -> str:
    return _float32_to_hex(float(mpmath.nstr(v, 8, strip_zeros=False)))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ────────────────────────────────────────────────────────────────────────────
#  FMA test vectors
# ────────────────────────────────────────────────────────────────────────────

def _generate_fma_vectors(count: int = 1024) -> list[dict[str, Any]]:
    """
    Generate FMA test vectors: expected = a * b + c at 128-bit precision.

    Categories (roughly count/4 each):
      - Normal range: uniform random in [-1e10, 1e10]
      - Denormals: values near FP32 minimum denormal (1.4e-45)
      - Exponent boundaries: powers of 2 near FP32 limits
      - Mantissa patterns: values with specific bit patterns
    """
    vectors: list[dict[str, Any]] = []
    quarter = count // 4

    # Use a fixed seed for reproducibility
    rng = mpmath.rand

    # Category 1: Normal range
    normal_values = [
        (1.0, 1.0, 0.0),
        (2.0, 3.0, 4.0),
        (-1.5, 2.5, 3.75),
        (1e6, 1e-6, 1.0),
        (0.1, 0.2, 0.3),
    ]
    for i in range(quarter):
        if i < len(normal_values):
            a, b, c = normal_values[i]
        else:
            # Deterministic pseudo-random based on index
            a = float(mpmath.mpf(((i * 7919 + 104729) % 200001 - 100000)) / 100)
            b = float(mpmath.mpf(((i * 6271 + 72931) % 200001 - 100000)) / 100)
            c = float(mpmath.mpf(((i * 5381 + 15485863) % 200001 - 100000)) / 100)

        ma, mb, mc = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(c)
        expected = ma * mb + mc
        vectors.append({
            "category": "normal",
            "a": a, "b": b, "c": c,
            "a_hex": _float32_to_hex(a),
            "b_hex": _float32_to_hex(b),
            "c_hex": _float32_to_hex(c),
            "expected_f64": _mpf_to_float64(expected),
            "expected_hex": _float_to_hex(_mpf_to_float64(expected)),
        })

    # Category 2: Denormals
    denorm_base = mpmath.mpf("1.401298464324817e-45")  # FP32 min denormal
    for i in range(quarter):
        scale = mpmath.mpf(1 + i)
        a = denorm_base * scale
        b = mpmath.mpf(1 + (i % 10))
        c = denorm_base * mpmath.mpf(i % 5)
        expected = a * b + c
        vectors.append({
            "category": "denormal",
            "a": float(a), "b": float(b), "c": float(c),
            "a_hex": _float_to_hex(float(a)),
            "b_hex": _float_to_hex(float(b)),
            "c_hex": _float_to_hex(float(c)),
            "expected_f64": _mpf_to_float64(expected),
            "expected_hex": _float_to_hex(_mpf_to_float64(expected)),
        })

    # Category 3: Exponent boundaries
    for i in range(quarter):
        exp = -126 + (i % 253)  # span FP32 exponent range
        a = mpmath.power(2, exp)
        b = mpmath.power(2, -exp // 2)
        c = mpmath.power(2, exp - 1) if exp > -126 else mpmath.mpf(0)
        expected = a * b + c
        vectors.append({
            "category": "exponent_boundary",
            "a": float(a), "b": float(b), "c": float(c),
            "a_hex": _float_to_hex(float(a)),
            "b_hex": _float_to_hex(float(b)),
            "c_hex": _float_to_hex(float(c)),
            "expected_f64": _mpf_to_float64(expected),
            "expected_hex": _float_to_hex(_mpf_to_float64(expected)),
        })

    # Category 4: Mantissa patterns (specific bit patterns)
    mantissa_patterns = [
        0x3F800000,  # 1.0
        0x3F800001,  # 1.0 + 1 ULP
        0x3FFFFFFF,  # ~2.0 - 1 ULP
        0x7F7FFFFF,  # FP32 max
        0x00800000,  # FP32 min normal
        0x00000001,  # FP32 min denormal
        0x3FC00000,  # 1.5
        0x3FE00000,  # 1.75
    ]
    for i in range(quarter):
        pat_idx = i % len(mantissa_patterns)
        a_bits = mantissa_patterns[pat_idx]
        a = struct.unpack("!f", struct.pack("!I", a_bits))[0]
        b_bits = mantissa_patterns[(pat_idx + 1) % len(mantissa_patterns)]
        b = struct.unpack("!f", struct.pack("!I", b_bits))[0]
        c = 0.0

        # Skip inf/nan
        if not (abs(a) < 1e38 and abs(b) < 1e38):
            a, b = 1.0, 1.0

        ma, mb, mc = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(c)
        expected = ma * mb + mc
        vectors.append({
            "category": "mantissa_pattern",
            "a": a, "b": b, "c": c,
            "a_hex": _float32_to_hex(a),
            "b_hex": _float32_to_hex(b),
            "c_hex": _float32_to_hex(c),
            "expected_f64": _mpf_to_float64(expected),
            "expected_hex": _float_to_hex(_mpf_to_float64(expected)),
        })

    return vectors[:count]


# ────────────────────────────────────────────────────────────────────────────
#  Tensor Core test cases
# ────────────────────────────────────────────────────────────────────────────

def _matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Arbitrary-precision 16x16 matrix multiply."""
    n = len(a)
    result = [[mpmath.mpf(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = mpmath.mpf(0)
            for k in range(n):
                s += mpmath.mpf(a[i][k]) * mpmath.mpf(b[k][j])
            result[i][j] = s
    return [[float(result[i][j]) for j in range(n)] for i in range(n)]


def _identity_matrix(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _permutation_matrix(n: int) -> list[list[float]]:
    """Cyclic permutation matrix."""
    m = [[0.0] * n for _ in range(n)]
    for i in range(n):
        m[i][(i + 1) % n] = 1.0
    return m


def _hadamard_matrix(n: int) -> list[list[float]]:
    """Sylvester-type Hadamard matrix (n must be power of 2), normalized."""
    if n == 1:
        return [[1.0]]
    half = _hadamard_matrix(n // 2)
    m = [[0.0] * n for _ in range(n)]
    for i in range(n // 2):
        for j in range(n // 2):
            m[i][j] = half[i][j]
            m[i][j + n // 2] = half[i][j]
            m[i + n // 2][j] = half[i][j]
            m[i + n // 2][j + n // 2] = -half[i][j]
    # Normalize
    scale = 1.0 / (n ** 0.5)
    return [[m[i][j] * scale for j in range(n)] for i in range(n)]


def _generate_tensor_core_cases() -> list[dict[str, Any]]:
    n = 16
    cases = []

    # Identity * Identity = Identity
    ident = _identity_matrix(n)
    cases.append({
        "name": "identity_x_identity",
        "a": ident,
        "b": ident,
        "expected": _matrix_multiply(ident, ident),
    })

    # Identity * Permutation = Permutation
    perm = _permutation_matrix(n)
    cases.append({
        "name": "identity_x_permutation",
        "a": ident,
        "b": perm,
        "expected": _matrix_multiply(ident, perm),
    })

    # Permutation * Permutation^T = Identity
    perm_t = [[perm[j][i] for j in range(n)] for i in range(n)]
    cases.append({
        "name": "permutation_x_transpose",
        "a": perm,
        "b": perm_t,
        "expected": _matrix_multiply(perm, perm_t),
    })

    # Hadamard * Hadamard^T = Identity (orthonormal)
    had = _hadamard_matrix(n)
    had_t = [[had[j][i] for j in range(n)] for i in range(n)]
    cases.append({
        "name": "hadamard_x_transpose",
        "a": had,
        "b": had_t,
        "expected": _matrix_multiply(had, had_t),
    })

    # Deterministic "random" matrix
    det_a = [
        [float(((i * 17 + j * 31 + 7) % 201 - 100) / 100.0)
         for j in range(n)]
        for i in range(n)
    ]
    det_b = [
        [float(((i * 13 + j * 37 + 11) % 201 - 100) / 100.0)
         for j in range(n)]
        for i in range(n)
    ]
    cases.append({
        "name": "deterministic_random",
        "a": det_a,
        "b": det_b,
        "expected": _matrix_multiply(det_a, det_b),
    })

    return cases


# ────────────────────────────────────────────────────────────────────────────
#  Transcendental function test vectors
# ────────────────────────────────────────────────────────────────────────────

def _generate_transcendental_vectors(per_function: int = 256) -> dict[str, list[dict[str, Any]]]:
    functions = {
        "sin": mpmath.sin,
        "cos": mpmath.cos,
        "exp": mpmath.exp,
        "log": mpmath.log,
        "rsqrt": lambda x: 1 / mpmath.sqrt(x),
    }

    results: dict[str, list[dict[str, Any]]] = {}

    for fname, func in functions.items():
        vectors: list[dict[str, Any]] = []

        for i in range(per_function):
            if fname in ("sin", "cos"):
                # Span [-4pi, 4pi] with concentration near 0 and pi
                t = mpmath.mpf(i) / per_function
                x = mpmath.mpf(-4) * mpmath.pi + t * 8 * mpmath.pi
            elif fname == "exp":
                # Span [-20, 20] to cover underflow/overflow for FP32
                x = mpmath.mpf(-20) + mpmath.mpf(i) * 40 / per_function
            elif fname == "log":
                # Span (1e-30, 1e30) on log scale
                x = mpmath.power(10, mpmath.mpf(-30) + mpmath.mpf(i) * 60 / per_function)
            elif fname == "rsqrt":
                # Span (1e-20, 1e20) — must be positive
                x = mpmath.power(10, mpmath.mpf(-20) + mpmath.mpf(i) * 40 / per_function)
            else:
                x = mpmath.mpf(i + 1)

            try:
                expected = func(x)
                vectors.append({
                    "input": float(x),
                    "input_hex": _float_to_hex(float(x)),
                    "expected_f64": float(expected),
                    "expected_hex": _float_to_hex(float(expected)),
                })
            except (ValueError, ZeroDivisionError):
                continue

        results[fname] = vectors

    return results


# ────────────────────────────────────────────────────────────────────────────
#  AES test vectors
# ────────────────────────────────────────────────────────────────────────────

def _generate_aes_vectors() -> dict[str, Any]:
    """
    Generate AES-128-ECB test vectors.  Uses a well-known NIST test key and
    extends to 4KB of plaintext with a deterministic pattern.
    """
    # NIST AES-128 test key
    key_hex = "2b7e151628aed2a6abf7158809cf4f3c"
    key = bytes.fromhex(key_hex)

    # Generate 4KB of deterministic plaintext
    plaintext = bytes([(i * 179 + 83) % 256 for i in range(4096)])

    # We store the reference — actual AES computation is verified against
    # hardware.  For the golden file we include the NIST known-answer test.
    nist_plaintext = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    nist_ciphertext = bytes.fromhex("3ad77bb40d7a3660a89ecaf32466ef97")

    return {
        "algorithm": "AES-128-ECB",
        "key_hex": key_hex,
        "plaintext_4kb_hex": plaintext.hex(),
        "plaintext_4kb_sha256": _sha256(plaintext),
        "nist_test_vector": {
            "plaintext_hex": nist_plaintext.hex(),
            "expected_ciphertext_hex": nist_ciphertext.hex(),
        },
        "block_count": len(plaintext) // 16,
    }


# ────────────────────────────────────────────────────────────────────────────
#  Memory test patterns (March C-)
# ────────────────────────────────────────────────────────────────────────────

def _generate_memory_patterns() -> list[dict[str, Any]]:
    """
    Generate data patterns for March C- memory test.

    March C- algorithm:
      M0: write 0 to all cells (ascending)
      M1: read 0, write 1 (ascending)
      M2: read 1, write 0 (ascending)
      M3: read 0, write 1 (descending)
      M4: read 1, write 0 (descending)
      M5: read 0 (ascending)
    """
    patterns = []

    # Standard patterns
    test_sizes = [256, 1024, 4096]
    for size in test_sizes:
        # All-zeros
        zeros = "00" * size
        # All-ones
        ones = "ff" * size
        # Checkerboard
        checker_a = ("aa" * (size // 2)) if size >= 2 else "aa"
        checker_b = ("55" * (size // 2)) if size >= 2 else "55"
        # Walking ones
        walking = bytes([(1 << (i % 8)) for i in range(size)]).hex()
        # Walking zeros
        walking_z = bytes([~(1 << (i % 8)) & 0xFF for i in range(size)]).hex()

        patterns.append({
            "name": f"march_c_minus_{size}B",
            "size_bytes": size,
            "patterns": {
                "all_zeros": zeros,
                "all_ones": ones,
                "checkerboard_a": checker_a,
                "checkerboard_b": checker_b,
                "walking_ones": walking,
                "walking_zeros": walking_z,
            },
            "march_c_steps": [
                {"step": "M0", "direction": "ascending", "write": "0x00"},
                {"step": "M1", "direction": "ascending", "read": "0x00", "write": "0xFF"},
                {"step": "M2", "direction": "ascending", "read": "0xFF", "write": "0x00"},
                {"step": "M3", "direction": "descending", "read": "0x00", "write": "0xFF"},
                {"step": "M4", "direction": "descending", "read": "0xFF", "write": "0x00"},
                {"step": "M5", "direction": "ascending", "read": "0x00"},
            ],
        })

    return patterns


# ────────────────────────────────────────────────────────────────────────────
#  Architecture-specific configuration
# ────────────────────────────────────────────────────────────────────────────

ARCH_CONFIG = {
    "ampere": {
        "compute_capability": "8.0",
        "fp_formats": ["fp32", "fp16", "bf16", "tf32"],
        "tensor_core_shape": "16x16x16",
        "sm_count_typical": 108,
    },
    "hopper": {
        "compute_capability": "9.0",
        "fp_formats": ["fp32", "fp16", "bf16", "tf32", "fp8_e4m3", "fp8_e5m2"],
        "tensor_core_shape": "16x16x16",
        "sm_count_typical": 132,
    },
    "ada": {
        "compute_capability": "8.9",
        "fp_formats": ["fp32", "fp16", "bf16", "tf32", "fp8_e4m3", "fp8_e5m2"],
        "tensor_core_shape": "16x16x16",
        "sm_count_typical": 128,
    },
    "blackwell": {
        "compute_capability": "10.0",
        "fp_formats": ["fp32", "fp16", "bf16", "tf32", "fp8_e4m3", "fp8_e5m2", "fp4"],
        "tensor_core_shape": "16x16x16",
        "sm_count_typical": 192,
    },
}


# ────────────────────────────────────────────────────────────────────────────
#  Main generation
# ────────────────────────────────────────────────────────────────────────────

def generate_golden_answers(arch: str) -> dict[str, Any]:
    """Generate a complete golden answer file for the given architecture."""
    if arch not in ARCH_CONFIG:
        raise ValueError(f"Unknown architecture: {arch!r}. "
                         f"Available: {list(ARCH_CONFIG.keys())}")

    print(f"Generating golden answers for {arch}...")

    print("  FMA vectors (1024)...")
    fma = _generate_fma_vectors(1024)

    print("  Tensor Core cases (16x16)...")
    tc = _generate_tensor_core_cases()

    print("  Transcendental vectors (256/function)...")
    trans = _generate_transcendental_vectors(256)

    print("  AES vectors (4KB)...")
    aes = _generate_aes_vectors()

    print("  Memory patterns (March C-)...")
    mem = _generate_memory_patterns()

    golden = {
        "metadata": {
            "version": "1.0.0",
            "architecture": arch,
            "architecture_config": ARCH_CONFIG[arch],
            "generation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "precision_bits": 128,
            "generator": "sentinel-golden-answer-generator",
        },
        "fma": {
            "description": "Fused Multiply-Add test vectors at 128-bit precision",
            "count": len(fma),
            "vectors": fma,
        },
        "tensor_core": {
            "description": "16x16 matrix multiply test cases for tensor core validation",
            "matrix_size": 16,
            "cases": tc,
        },
        "transcendental": {
            "description": "Transcendental function test vectors (sin, cos, exp, log, rsqrt)",
            "functions": trans,
        },
        "aes": {
            "description": "AES-128-ECB test vectors for crypto unit validation",
            **aes,
        },
        "memory": {
            "description": "Memory test patterns for March C- algorithm",
            "patterns": mem,
        },
    }

    print(f"  Done. Total FMA vectors: {len(fma)}")
    return golden


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate golden reference answers for SENTINEL probe agent"
    )
    parser.add_argument(
        "--arch",
        choices=ARCHITECTURES,
        help="Target GPU architecture family",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all architectures",
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path (used with --arch)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (used with --all)",
    )
    args = parser.parse_args()

    if not args.arch and not args.all:
        parser.error("Specify --arch <name> or --all")

    if args.all:
        out_dir = Path(args.output_dir or "golden")
        out_dir.mkdir(parents=True, exist_ok=True)
        for arch in ARCHITECTURES:
            golden = generate_golden_answers(arch)
            out_path = out_dir / f"golden_{arch}.json"
            out_path.write_text(
                json.dumps(golden, indent=2), encoding="utf-8"
            )
            print(f"Written: {out_path}")
    else:
        golden = generate_golden_answers(args.arch)
        out_path = args.output or f"golden_{args.arch}.json"
        Path(out_path).write_text(
            json.dumps(golden, indent=2), encoding="utf-8"
        )
        print(f"Written: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
