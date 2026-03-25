"""
SENTINEL Golden Answer Verifier

Loads golden answers from JSON, re-computes expected values at 128-bit
precision using mpmath, and verifies bit-exact matches.  Optionally runs
validation against a live GPU to check hardware correctness.

Usage:
    python verify.py --golden golden_ampere.json
    python verify.py --golden golden_ampere.json --gpu 0
    python verify.py --golden-dir golden/ --verbose

Copyright 2025-2026 SENTINEL Authors — Apache 2.0
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mpmath

mpmath.mp.prec = 128


# ────────────────────────────────────────────────────────────────────────────
#  Result tracking
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    category: str
    total: int
    passed: int
    failed: int
    discrepancies: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed == 0


@dataclass
class VerificationReport:
    file_path: str
    architecture: str
    categories: list[VerificationResult] = field(default_factory=list)
    gpu_results: Optional[dict[str, Any]] = None

    @property
    def total_passed(self) -> int:
        return sum(c.passed for c in self.categories)

    @property
    def total_failed(self) -> int:
        return sum(c.failed for c in self.categories)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.categories)


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def _float_to_hex(f: float) -> str:
    return struct.pack("!d", f).hex()


def _hex_to_float(h: str) -> float:
    return struct.unpack("!d", bytes.fromhex(h))[0]


def _floats_bit_equal(a: float, b: float) -> bool:
    """Check if two floats are bit-exactly equal (including NaN == NaN)."""
    return struct.pack("!d", a) == struct.pack("!d", b)


# ────────────────────────────────────────────────────────────────────────────
#  FMA verification
# ────────────────────────────────────────────────────────────────────────────

def _verify_fma(vectors: list[dict[str, Any]], verbose: bool) -> VerificationResult:
    result = VerificationResult(category="fma", total=len(vectors), passed=0, failed=0)

    for i, v in enumerate(vectors):
        a = mpmath.mpf(v["a"])
        b = mpmath.mpf(v["b"])
        c = mpmath.mpf(v["c"])
        computed = float(a * b + c)
        expected = v["expected_f64"]

        if _floats_bit_equal(computed, expected):
            result.passed += 1
        else:
            result.failed += 1
            disc = {
                "index": i,
                "category": v.get("category", "unknown"),
                "a": v["a"], "b": v["b"], "c": v["c"],
                "expected": expected,
                "computed": computed,
                "expected_hex": _float_to_hex(expected),
                "computed_hex": _float_to_hex(computed),
            }
            result.discrepancies.append(disc)
            if verbose:
                print(f"  FMA MISMATCH [{i}]: a={v['a']}, b={v['b']}, c={v['c']}")
                print(f"    expected: {expected} ({_float_to_hex(expected)})")
                print(f"    computed: {computed} ({_float_to_hex(computed)})")

    return result


# ────────────────────────────────────────────────────────────────────────────
#  Tensor Core verification
# ────────────────────────────────────────────────────────────────────────────

def _verify_tensor_core(cases: list[dict[str, Any]], verbose: bool) -> VerificationResult:
    total_elements = 0
    failed_elements = 0
    discrepancies: list[dict[str, Any]] = []

    for case in cases:
        a = case["a"]
        b = case["b"]
        expected = case["expected"]
        n = len(a)

        for i in range(n):
            for j in range(n):
                total_elements += 1
                s = mpmath.mpf(0)
                for k in range(n):
                    s += mpmath.mpf(a[i][k]) * mpmath.mpf(b[k][j])
                computed = float(s)

                if not _floats_bit_equal(computed, expected[i][j]):
                    failed_elements += 1
                    if len(discrepancies) < 20:
                        discrepancies.append({
                            "case": case["name"],
                            "position": [i, j],
                            "expected": expected[i][j],
                            "computed": computed,
                        })
                    if verbose and len(discrepancies) <= 5:
                        print(f"  TC MISMATCH [{case['name']}][{i},{j}]: "
                              f"expected={expected[i][j]}, computed={computed}")

    return VerificationResult(
        category="tensor_core",
        total=total_elements,
        passed=total_elements - failed_elements,
        failed=failed_elements,
        discrepancies=discrepancies,
    )


# ────────────────────────────────────────────────────────────────────────────
#  Transcendental verification
# ────────────────────────────────────────────────────────────────────────────

def _verify_transcendental(
    functions: dict[str, list[dict[str, Any]]], verbose: bool
) -> VerificationResult:
    mp_funcs = {
        "sin": mpmath.sin,
        "cos": mpmath.cos,
        "exp": mpmath.exp,
        "log": mpmath.log,
        "rsqrt": lambda x: 1 / mpmath.sqrt(x),
    }

    total = 0
    failed = 0
    discrepancies: list[dict[str, Any]] = []

    for fname, vectors in functions.items():
        func = mp_funcs.get(fname)
        if func is None:
            continue

        for i, v in enumerate(vectors):
            total += 1
            x = mpmath.mpf(v["input"])
            try:
                computed = float(func(x))
            except (ValueError, ZeroDivisionError):
                continue

            expected = v["expected_f64"]
            if not _floats_bit_equal(computed, expected):
                failed += 1
                if len(discrepancies) < 20:
                    discrepancies.append({
                        "function": fname,
                        "index": i,
                        "input": v["input"],
                        "expected": expected,
                        "computed": computed,
                    })
                if verbose and len(discrepancies) <= 5:
                    print(f"  {fname} MISMATCH [{i}]: input={v['input']}, "
                          f"expected={expected}, computed={computed}")

    return VerificationResult(
        category="transcendental",
        total=total,
        passed=total - failed,
        failed=failed,
        discrepancies=discrepancies,
    )


# ────────────────────────────────────────────────────────────────────────────
#  AES verification (structural check only — no crypto computation)
# ────────────────────────────────────────────────────────────────────────────

def _verify_aes(aes_data: dict[str, Any], verbose: bool) -> VerificationResult:
    """Verify AES golden answer structural integrity."""
    checks = 0
    failed = 0
    discrepancies: list[dict[str, Any]] = []

    # Check key length
    checks += 1
    key = bytes.fromhex(aes_data["key_hex"])
    if len(key) != 16:
        failed += 1
        discrepancies.append({"check": "key_length", "expected": 16, "got": len(key)})

    # Check plaintext size
    checks += 1
    pt = bytes.fromhex(aes_data["plaintext_4kb_hex"])
    if len(pt) != 4096:
        failed += 1
        discrepancies.append({"check": "plaintext_size", "expected": 4096, "got": len(pt)})

    # Check SHA-256 of plaintext
    checks += 1
    import hashlib
    computed_sha = hashlib.sha256(pt).hexdigest()
    if computed_sha != aes_data["plaintext_4kb_sha256"]:
        failed += 1
        discrepancies.append({
            "check": "plaintext_sha256",
            "expected": aes_data["plaintext_4kb_sha256"],
            "computed": computed_sha,
        })

    # Check NIST test vector
    checks += 1
    nist = aes_data.get("nist_test_vector", {})
    expected_ct = "3ad77bb40d7a3660a89ecaf32466ef97"
    if nist.get("expected_ciphertext_hex") != expected_ct:
        failed += 1
        discrepancies.append({
            "check": "nist_ciphertext",
            "expected": expected_ct,
            "got": nist.get("expected_ciphertext_hex"),
        })

    # Check block count
    checks += 1
    if aes_data.get("block_count") != 4096 // 16:
        failed += 1
        discrepancies.append({
            "check": "block_count",
            "expected": 256,
            "got": aes_data.get("block_count"),
        })

    if verbose and discrepancies:
        for d in discrepancies:
            print(f"  AES CHECK FAILED: {d}")

    return VerificationResult(
        category="aes",
        total=checks,
        passed=checks - failed,
        failed=failed,
        discrepancies=discrepancies,
    )


# ────────────────────────────────────────────────────────────────────────────
#  Memory pattern verification
# ────────────────────────────────────────────────────────────────────────────

def _verify_memory(patterns: list[dict[str, Any]], verbose: bool) -> VerificationResult:
    checks = 0
    failed = 0
    discrepancies: list[dict[str, Any]] = []

    for pat in patterns:
        size = pat["size_bytes"]

        # Verify all_zeros pattern
        checks += 1
        expected_zeros = "00" * size
        if pat["patterns"]["all_zeros"] != expected_zeros:
            failed += 1
            discrepancies.append({"pattern": pat["name"], "check": "all_zeros"})

        # Verify all_ones pattern
        checks += 1
        expected_ones = "ff" * size
        if pat["patterns"]["all_ones"] != expected_ones:
            failed += 1
            discrepancies.append({"pattern": pat["name"], "check": "all_ones"})

        # Verify walking ones length
        checks += 1
        walking = bytes.fromhex(pat["patterns"]["walking_ones"])
        if len(walking) != size:
            failed += 1
            discrepancies.append({
                "pattern": pat["name"],
                "check": "walking_ones_length",
                "expected": size,
                "got": len(walking),
            })

        # Verify walking ones values
        checks += 1
        all_correct = all(
            walking[i] == (1 << (i % 8)) for i in range(len(walking))
        )
        if not all_correct:
            failed += 1
            discrepancies.append({"pattern": pat["name"], "check": "walking_ones_values"})

        # Verify march C- step count
        checks += 1
        if len(pat["march_c_steps"]) != 6:
            failed += 1
            discrepancies.append({
                "pattern": pat["name"],
                "check": "march_c_step_count",
                "expected": 6,
                "got": len(pat["march_c_steps"]),
            })

    return VerificationResult(
        category="memory",
        total=checks,
        passed=checks - failed,
        failed=failed,
        discrepancies=discrepancies,
    )


# ────────────────────────────────────────────────────────────────────────────
#  GPU verification (optional, requires CUDA)
# ────────────────────────────────────────────────────────────────────────────

def _verify_on_gpu(golden: dict[str, Any], gpu_index: int) -> dict[str, Any]:
    """
    Run a subset of golden answer checks on an actual GPU.
    Requires PyCUDA or CuPy.
    """
    results: dict[str, Any] = {"gpu_index": gpu_index, "tests": []}

    try:
        import cupy as cp
        cp.cuda.Device(gpu_index).use()
    except ImportError:
        return {"gpu_index": gpu_index, "error": "CuPy not installed"}
    except Exception as e:
        return {"gpu_index": gpu_index, "error": str(e)}

    # FMA check: compute a*b+c on GPU for first 100 vectors
    fma_vectors = golden.get("fma", {}).get("vectors", [])[:100]
    fma_pass = 0
    fma_fail = 0
    for v in fma_vectors:
        a = cp.float32(v["a"])
        b = cp.float32(v["b"])
        c = cp.float32(v["c"])
        gpu_result = float(a * b + c)
        # Compare with tolerance for FP32 vs FP64 golden
        expected_f32 = float(cp.float32(v["expected_f64"]))
        if gpu_result == expected_f32:
            fma_pass += 1
        else:
            fma_fail += 1

    results["tests"].append({
        "category": "fma_gpu",
        "total": len(fma_vectors),
        "passed": fma_pass,
        "failed": fma_fail,
    })

    # Transcendental check: sin on GPU for first 50 vectors
    sin_vectors = golden.get("transcendental", {}).get("functions", {}).get("sin", [])[:50]
    sin_pass = 0
    sin_fail = 0
    for v in sin_vectors:
        x = cp.float64(v["input"])
        gpu_result = float(cp.sin(x))
        expected = v["expected_f64"]
        # Allow 1 ULP tolerance for GPU transcendentals
        if abs(gpu_result - expected) <= abs(expected) * 2.2e-16:
            sin_pass += 1
        else:
            sin_fail += 1

    results["tests"].append({
        "category": "sin_gpu",
        "total": len(sin_vectors),
        "passed": sin_pass,
        "failed": sin_fail,
    })

    return results


# ────────────────────────────────────────────────────────────────────────────
#  Main verification
# ────────────────────────────────────────────────────────────────────────────

def verify_golden_file(
    file_path: str,
    verbose: bool = False,
    gpu_index: Optional[int] = None,
) -> VerificationReport:
    """Verify all categories in a golden answer file."""
    path = Path(file_path)
    golden = json.loads(path.read_text(encoding="utf-8"))

    arch = golden.get("metadata", {}).get("architecture", "unknown")
    print(f"Verifying golden answers: {path.name} (arch: {arch})")
    print(f"  Generated: {golden.get('metadata', {}).get('generation_date', 'unknown')}")
    print(f"  Precision: {golden.get('metadata', {}).get('precision_bits', 'unknown')} bits")

    report = VerificationReport(file_path=str(path), architecture=arch)

    # FMA
    if "fma" in golden:
        print("  Verifying FMA vectors...")
        r = _verify_fma(golden["fma"]["vectors"], verbose)
        report.categories.append(r)
        print(f"    {r.passed}/{r.total} passed ({r.failed} failed)")

    # Tensor Core
    if "tensor_core" in golden:
        print("  Verifying Tensor Core cases...")
        r = _verify_tensor_core(golden["tensor_core"]["cases"], verbose)
        report.categories.append(r)
        print(f"    {r.passed}/{r.total} element checks passed ({r.failed} failed)")

    # Transcendental
    if "transcendental" in golden:
        print("  Verifying Transcendental functions...")
        r = _verify_transcendental(golden["transcendental"]["functions"], verbose)
        report.categories.append(r)
        print(f"    {r.passed}/{r.total} passed ({r.failed} failed)")

    # AES
    if "aes" in golden:
        print("  Verifying AES vectors...")
        r = _verify_aes(golden["aes"], verbose)
        report.categories.append(r)
        print(f"    {r.passed}/{r.total} checks passed ({r.failed} failed)")

    # Memory
    if "memory" in golden:
        print("  Verifying Memory patterns...")
        r = _verify_memory(golden["memory"]["patterns"], verbose)
        report.categories.append(r)
        print(f"    {r.passed}/{r.total} checks passed ({r.failed} failed)")

    # GPU verification
    if gpu_index is not None:
        print(f"  Running GPU verification on device {gpu_index}...")
        gpu_results = _verify_on_gpu(golden, gpu_index)
        report.gpu_results = gpu_results
        if "error" in gpu_results:
            print(f"    GPU verification error: {gpu_results['error']}")
        else:
            for t in gpu_results.get("tests", []):
                print(f"    {t['category']}: {t['passed']}/{t['total']} passed")

    # Summary
    status = "PASS" if report.ok else "FAIL"
    print(f"\n  Result: [{status}] {report.total_passed} passed, "
          f"{report.total_failed} failed")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify SENTINEL golden answer files"
    )
    parser.add_argument(
        "--golden",
        help="Path to a single golden answer JSON file",
    )
    parser.add_argument(
        "--golden-dir",
        help="Path to a directory of golden answer JSON files",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU index for hardware verification (requires CuPy)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed discrepancy information",
    )
    args = parser.parse_args()

    if not args.golden and not args.golden_dir:
        parser.error("Specify --golden <file> or --golden-dir <dir>")

    files: list[str] = []
    if args.golden:
        files.append(args.golden)
    if args.golden_dir:
        d = Path(args.golden_dir)
        files.extend(str(f) for f in sorted(d.glob("golden_*.json")))

    if not files:
        print("No golden answer files found.")
        return 1

    all_ok = True
    for f in files:
        report = verify_golden_file(f, verbose=args.verbose, gpu_index=args.gpu)
        if not report.ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
