"""Tests for the tensor fingerprint locality-sensitive hash."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel_inference.signatures.tensor_fingerprint import TensorFingerprint


class TestTensorFingerprint:
    def test_fingerprint_length(self) -> None:
        """Fingerprint should always be 16 bytes (128 bits)."""
        fp = TensorFingerprint(projection_dim=64, bits_per_dim=2, seed=42)
        tensor = np.random.randn(1000).astype(np.float32)
        result = fp.compute(tensor)
        assert len(result) == 16
        assert isinstance(result, bytes)

    def test_deterministic(self) -> None:
        """Same tensor with same seed should produce identical fingerprints."""
        fp = TensorFingerprint(seed=42)
        tensor = np.random.randn(500).astype(np.float32)
        fp1 = fp.compute(tensor)
        fp2 = fp.compute(tensor)
        assert fp1 == fp2

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different fingerprints."""
        tensor = np.random.randn(500).astype(np.float32)
        fp1 = TensorFingerprint(seed=42).compute(tensor)
        fp2 = TensorFingerprint(seed=99).compute(tensor)
        # They could theoretically be the same but extremely unlikely
        assert fp1 != fp2

    def test_similar_tensors_small_hamming(self) -> None:
        """Similar tensors should have small Hamming distance."""
        fp = TensorFingerprint(seed=42)
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, size=1000).astype(np.float32)
        perturbed = base + rng.normal(0, 0.01, size=1000).astype(np.float32)

        fp_base = fp.compute(base)
        fp_perturbed = fp.compute(perturbed)

        hamming = TensorFingerprint.hamming_distance(fp_base, fp_perturbed)
        # Small perturbation should produce small Hamming distance
        assert hamming < 40, f"Hamming distance {hamming} too large for small perturbation"

    def test_different_tensors_large_hamming(self) -> None:
        """Very different tensors should have larger Hamming distance."""
        fp = TensorFingerprint(seed=42)
        rng = np.random.default_rng(0)
        tensor_a = rng.normal(0, 1, size=1000).astype(np.float32)
        tensor_b = rng.normal(10, 5, size=1000).astype(np.float32)

        fp_a = fp.compute(tensor_a)
        fp_b = fp.compute(tensor_b)

        hamming = TensorFingerprint.hamming_distance(fp_a, fp_b)
        assert hamming > 10, f"Hamming distance {hamming} too small for different tensors"

    def test_hamming_distance_self_zero(self) -> None:
        fp = TensorFingerprint(seed=42)
        tensor = np.random.randn(100).astype(np.float32)
        result = fp.compute(tensor)
        assert TensorFingerprint.hamming_distance(result, result) == 0

    def test_hamming_distance_symmetric(self) -> None:
        fp = TensorFingerprint(seed=42)
        a = fp.compute(np.random.randn(100).astype(np.float32))
        b = fp.compute(np.random.randn(100).astype(np.float32))
        assert TensorFingerprint.hamming_distance(a, b) == TensorFingerprint.hamming_distance(b, a)

    def test_hamming_distance_bounds(self) -> None:
        fp = TensorFingerprint(seed=42)
        a = fp.compute(np.random.randn(100).astype(np.float32))
        b = fp.compute(np.random.randn(100).astype(np.float32))
        d = TensorFingerprint.hamming_distance(a, b)
        assert 0 <= d <= 128

    def test_hamming_distance_invalid_length(self) -> None:
        with pytest.raises(ValueError):
            TensorFingerprint.hamming_distance(b"\x00" * 15, b"\x00" * 16)

    def test_cosine_distance_estimate(self) -> None:
        """Cosine distance estimate should be in [0, 2]."""
        fp = TensorFingerprint(seed=42)
        a = fp.compute(np.random.randn(100).astype(np.float32))
        b = fp.compute(np.random.randn(100).astype(np.float32))
        cd = TensorFingerprint.cosine_distance_estimate(a, b)
        assert 0.0 <= cd <= 2.0

    def test_cosine_distance_self_zero(self) -> None:
        fp = TensorFingerprint(seed=42)
        tensor = np.random.randn(100).astype(np.float32)
        result = fp.compute(tensor)
        cd = TensorFingerprint.cosine_distance_estimate(result, result)
        assert abs(cd) < 1e-10

    def test_different_input_dimensions(self) -> None:
        """Should handle tensors of different sizes."""
        fp = TensorFingerprint(seed=42)
        fp_small = fp.compute(np.random.randn(50).astype(np.float32))
        fp_large = fp.compute(np.random.randn(10000).astype(np.float32))
        assert len(fp_small) == 16
        assert len(fp_large) == 16

    def test_2d_tensor(self) -> None:
        """Should flatten multi-dimensional tensors."""
        fp = TensorFingerprint(seed=42)
        tensor_2d = np.random.randn(10, 100).astype(np.float32)
        tensor_1d = tensor_2d.ravel()
        fp_2d = fp.compute(tensor_2d)
        fp_1d = fp.compute(tensor_1d)
        assert fp_2d == fp_1d
