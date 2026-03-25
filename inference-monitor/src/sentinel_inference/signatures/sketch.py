"""Count-Min Sketch and HyperLogLog for output vocabulary monitoring.

These probabilistic data structures enable efficient tracking of token
frequency distributions and vocabulary cardinality without storing full
histograms.  Distributional shifts in output tokens are an early
indicator of silent data corruption.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np


class CountMinSketch:
    """Count-Min Sketch for frequency estimation of output tokens.

    Provides approximate frequency counts with guaranteed error bounds:
    estimated_count >= true_count and
    estimated_count <= true_count + epsilon * N
    where epsilon = e / width and N = total count.

    Parameters
    ----------
    width : int
        Number of counters per row.
    depth : int
        Number of hash functions (rows).
    """

    def __init__(self, width: int = 2048, depth: int = 5) -> None:
        self._width = width
        self._depth = depth
        self._table = np.zeros((depth, width), dtype=np.int64)
        self._total: int = 0
        # Pre-compute hash seeds
        self._seeds = [
            int.from_bytes(
                hashlib.sha256(f"cms_seed_{i}".encode()).digest()[:8],
                byteorder="little",
            )
            for i in range(depth)
        ]

    @property
    def total(self) -> int:
        return self._total

    def _hash(self, item: int, row: int) -> int:
        """Hash an item to a column index for a given row."""
        # MurmurHash-style mixing
        h = item ^ self._seeds[row]
        h = ((h >> 16) ^ h) * 0x45D9F3B
        h = ((h >> 16) ^ h) * 0x45D9F3B
        h = (h >> 16) ^ h
        return h % self._width

    def add(self, item: int, count: int = 1) -> None:
        """Add an item (token ID) with the given count."""
        self._total += count
        for row in range(self._depth):
            col = self._hash(item, row)
            self._table[row, col] += count

    def estimate(self, item: int) -> int:
        """Estimate the frequency of an item.

        Returns the minimum count across all hash functions (conservative
        estimate with lowest overcount).
        """
        min_count = np.iinfo(np.int64).max
        for row in range(self._depth):
            col = self._hash(item, row)
            min_count = min(min_count, int(self._table[row, col]))
        return min_count

    def merge(self, other: "CountMinSketch") -> None:
        """Merge another CMS into this one (element-wise addition)."""
        if self._width != other._width or self._depth != other._depth:
            raise ValueError("Cannot merge sketches with different dimensions.")
        self._table += other._table
        self._total += other._total

    def reset(self) -> None:
        self._table[:] = 0
        self._total = 0

    def relative_error(self, item: int) -> float:
        """Estimated relative frequency of an item."""
        if self._total == 0:
            return 0.0
        return self.estimate(item) / self._total

    def detect_shift(
        self,
        other: "CountMinSketch",
        top_k_items: list[int],
        threshold: float = 0.05,
    ) -> list[tuple[int, float, float]]:
        """Detect frequency shifts between two sketches.

        Parameters
        ----------
        other : CountMinSketch
            The other sketch to compare against.
        top_k_items : list of int
            Token IDs to check.
        threshold : float
            Minimum relative frequency difference to flag.

        Returns
        -------
        list of (token_id, freq_self, freq_other) for shifted items.
        """
        shifts: list[tuple[int, float, float]] = []
        for item in top_k_items:
            freq_self = self.relative_error(item)
            freq_other = other.relative_error(item)
            if abs(freq_self - freq_other) > threshold:
                shifts.append((item, freq_self, freq_other))
        return shifts


class HyperLogLogSketch:
    """HyperLogLog for cardinality estimation of output vocabulary.

    Estimates the number of distinct tokens seen with ~1.04/sqrt(m) relative
    error where m = 2^precision.

    Parameters
    ----------
    precision : int
        Number of bits for register indexing (4-18).
        m = 2^precision registers.
    """

    def __init__(self, precision: int = 14) -> None:
        if not (4 <= precision <= 18):
            raise ValueError(f"Precision must be in [4, 18], got {precision}")
        self._precision = precision
        self._m = 1 << precision
        self._registers = np.zeros(self._m, dtype=np.uint8)
        # Alpha constant for bias correction
        if self._m == 16:
            self._alpha = 0.673
        elif self._m == 32:
            self._alpha = 0.697
        elif self._m == 64:
            self._alpha = 0.709
        else:
            self._alpha = 0.7213 / (1.0 + 1.079 / self._m)

    def _hash(self, item: int) -> int:
        """Hash an item to a 64-bit integer."""
        data = struct.pack("<q", item)
        h = hashlib.md5(data).digest()
        return struct.unpack("<Q", h[:8])[0]

    @staticmethod
    def _leading_zeros(value: int, max_bits: int) -> int:
        """Count leading zeros in the binary representation."""
        if value == 0:
            return max_bits
        count = 0
        for i in range(max_bits - 1, -1, -1):
            if value & (1 << i):
                break
            count += 1
        return count

    def add(self, item: int) -> None:
        """Add a token ID to the sketch."""
        h = self._hash(item)
        # Use first precision bits as register index
        idx = h & (self._m - 1)
        # Remaining bits for leading-zero count
        remaining = h >> self._precision
        rank = self._leading_zeros(remaining, 64 - self._precision) + 1
        self._registers[idx] = max(self._registers[idx], rank)

    def estimate(self) -> float:
        """Estimate the cardinality (number of distinct items).

        Uses the bias-corrected HyperLogLog estimator with small/large
        range corrections.
        """
        # Raw estimate
        harmonic_sum = np.sum(np.power(2.0, -self._registers.astype(np.float64)))
        raw_estimate = self._alpha * self._m * self._m / harmonic_sum

        # Small range correction
        if raw_estimate <= 2.5 * self._m:
            zeros = int(np.sum(self._registers == 0))
            if zeros > 0:
                return self._m * np.log(self._m / zeros)
            return raw_estimate

        # Large range correction (for 64-bit hash)
        two_to_64 = 2.0**64
        if raw_estimate > two_to_64 / 30.0:
            return -two_to_64 * np.log(1.0 - raw_estimate / two_to_64)

        return raw_estimate

    def merge(self, other: "HyperLogLogSketch") -> None:
        """Merge another HLL into this one (element-wise max)."""
        if self._precision != other._precision:
            raise ValueError("Cannot merge HLLs with different precision.")
        np.maximum(self._registers, other._registers, out=self._registers)

    def reset(self) -> None:
        self._registers[:] = 0
