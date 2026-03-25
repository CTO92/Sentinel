"""128-bit tensor fingerprint via random projection.

Generates a locality-sensitive hash of a tensor such that Hamming distance
between fingerprints approximates cosine distance between the original
tensors.  This enables efficient similarity search and change detection.

Algorithm:
1. Generate a fixed random projection matrix R of shape
   (projection_dim, tensor_dim) using a deterministic seed.
2. Project the flattened tensor: z = R @ x  (shape: projection_dim)
3. Quantize each projected dimension to ``bits_per_dim`` bits using
   uniform quantization over the observed range.
4. Pack bits into a 128-bit (16-byte) fingerprint.

With projection_dim=64 and bits_per_dim=2, we get 64*2 = 128 bits.
"""

from __future__ import annotations

import struct

import numpy as np


class TensorFingerprint:
    """Locality-sensitive 128-bit fingerprint for tensors.

    Parameters
    ----------
    projection_dim : int
        Number of random projection dimensions (default 64).
    bits_per_dim : int
        Quantization bits per dimension (default 2).
    seed : int
        Fixed random seed for reproducible projections.
    """

    def __init__(
        self,
        projection_dim: int = 64,
        bits_per_dim: int = 2,
        seed: int = 42,
    ) -> None:
        self._projection_dim = projection_dim
        self._bits_per_dim = bits_per_dim
        self._seed = seed
        self._total_bits = projection_dim * bits_per_dim
        # Projection matrices are lazily initialized per input dimension
        self._projections: dict[int, np.ndarray] = {}
        self._rng = np.random.RandomState(seed)  # Use legacy for reproducibility

    def _get_projection(self, input_dim: int) -> np.ndarray:
        """Get or create the random projection matrix for a given input dim."""
        if input_dim not in self._projections:
            # Use a child RNG seeded deterministically from input_dim
            rng = np.random.RandomState(self._seed ^ input_dim)
            # Gaussian random projection (preserves cosine similarity)
            proj = rng.randn(self._projection_dim, input_dim).astype(np.float32)
            # Normalize rows for unit-norm projections
            norms = np.linalg.norm(proj, axis=1, keepdims=True)
            proj /= np.maximum(norms, 1e-10)
            self._projections[input_dim] = proj
        return self._projections[input_dim]

    def compute(self, tensor: np.ndarray) -> bytes:
        """Compute a 128-bit fingerprint of the given tensor.

        Parameters
        ----------
        tensor : np.ndarray
            Input tensor of any shape.

        Returns
        -------
        bytes
            16-byte fingerprint (128 bits).
        """
        flat = tensor.astype(np.float32).ravel()
        proj_matrix = self._get_projection(len(flat))

        # Project
        projected = proj_matrix @ flat  # shape: (projection_dim,)

        # Quantize each dimension to bits_per_dim bits
        # Map values to [0, 2^bits_per_dim - 1] using percentile-based binning
        num_levels = (1 << self._bits_per_dim) - 1
        # Use sign-magnitude quantization: map to [0, num_levels]
        p_min = float(np.min(projected))
        p_max = float(np.max(projected))
        if p_max - p_min < 1e-10:
            quantized = np.zeros(self._projection_dim, dtype=np.uint8)
        else:
            normalized = (projected - p_min) / (p_max - p_min)
            quantized = np.clip(
                np.round(normalized * num_levels).astype(np.uint8),
                0,
                num_levels,
            )

        # Pack bits into bytes
        return self._pack_bits(quantized)

    def _pack_bits(self, quantized: np.ndarray) -> bytes:
        """Pack quantized values into a byte string.

        Each value uses ``bits_per_dim`` bits, packed MSB-first.
        Result is padded to 16 bytes (128 bits).
        """
        bits: list[int] = []
        for val in quantized:
            for bit_pos in range(self._bits_per_dim - 1, -1, -1):
                bits.append(int((val >> bit_pos) & 1))

        # Pad to 128 bits
        while len(bits) < 128:
            bits.append(0)
        bits = bits[:128]

        # Pack into 16 bytes
        result = bytearray(16)
        for i, bit in enumerate(bits):
            byte_idx = i // 8
            bit_idx = 7 - (i % 8)
            if bit:
                result[byte_idx] |= 1 << bit_idx

        return bytes(result)

    @staticmethod
    def hamming_distance(fp1: bytes, fp2: bytes) -> int:
        """Compute Hamming distance between two fingerprints.

        Parameters
        ----------
        fp1, fp2 : bytes
            16-byte fingerprints.

        Returns
        -------
        int
            Number of differing bits (0 to 128).
        """
        if len(fp1) != 16 or len(fp2) != 16:
            raise ValueError("Fingerprints must be 16 bytes each.")

        # Unpack as two 64-bit integers for fast popcount
        a_hi, a_lo = struct.unpack(">QQ", fp1)
        b_hi, b_lo = struct.unpack(">QQ", fp2)

        xor_hi = a_hi ^ b_hi
        xor_lo = a_lo ^ b_lo

        return bin(xor_hi).count("1") + bin(xor_lo).count("1")

    @staticmethod
    def cosine_distance_estimate(fp1: bytes, fp2: bytes) -> float:
        """Estimate cosine distance from Hamming distance.

        For random-projection LSH, Hamming distance / total_bits
        approximates the angle between vectors / pi, so:

            cos_distance approx 1 - cos(pi * hamming / total_bits)

        Returns a value in [0, 2].
        """
        hamming = TensorFingerprint.hamming_distance(fp1, fp2)
        angle = np.pi * hamming / 128.0
        return float(1.0 - np.cos(angle))
