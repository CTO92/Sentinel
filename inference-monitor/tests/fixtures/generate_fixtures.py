"""Generate .npy fixture files for SENTINEL inference monitor tests.

Creates two fixture files:
- normal_logits.npy: 100 samples of well-behaved logit vectors drawn from
  N(0, 1), simulating healthy inference outputs.
- corrupted_logits.npy: 100 samples with SDC-like corruption -- mean-shifted,
  sporadic bit-flip spikes, and entropy collapse patterns.

Usage:
    python generate_fixtures.py
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def generate_normal_logits(
    num_samples: int = 100,
    vocab_size: int = 32000,
    seed: int = 42,
) -> np.ndarray:
    """Generate well-behaved logit vectors from N(0, 1).

    Returns shape (num_samples, vocab_size) float32 array.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(num_samples, vocab_size)).astype(np.float32)


def generate_corrupted_logits(
    num_samples: int = 100,
    vocab_size: int = 32000,
    seed: int = 42,
) -> np.ndarray:
    """Generate logit vectors exhibiting various SDC corruption patterns.

    Corruption types applied:
    1. Mean shift (first 30 samples): logits shifted by +5 sigma.
    2. Bit-flip spikes (next 30 samples): random positions have values
       replaced with very large magnitudes, simulating HBM bit flips.
    3. Entropy collapse (next 20 samples): one logit set extremely high,
       all others suppressed.
    4. Random noise (last 20 samples): uniform random replacing normal
       distribution, simulating SRAM corruption in compute units.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.0, size=(num_samples, vocab_size)).astype(np.float32)

    # --- Mean shift corruption (samples 0-29) ---
    logits[:30] += 5.0

    # --- Bit-flip spikes (samples 30-59) ---
    for i in range(30, 60):
        num_flips = rng.integers(1, 20)
        flip_positions = rng.choice(vocab_size, size=num_flips, replace=False)
        for pos in flip_positions:
            # Simulate a bit flip in float32 representation
            raw_bytes = logits[i, pos].tobytes()
            val = struct.unpack("<I", raw_bytes)[0]
            # Flip a random bit in the mantissa or exponent
            bit_to_flip = int(rng.integers(0, 32))
            val ^= 1 << bit_to_flip
            flipped = struct.unpack("<f", struct.pack("<I", val))[0]
            if np.isfinite(flipped):
                logits[i, pos] = flipped
            else:
                logits[i, pos] = 1e6 * (1 if rng.random() > 0.5 else -1)

    # --- Entropy collapse (samples 60-79) ---
    for i in range(60, 80):
        logits[i, :] = -100.0
        peak_pos = int(rng.integers(0, vocab_size))
        logits[i, peak_pos] = 100.0

    # --- Random noise (samples 80-99) ---
    logits[80:] = rng.uniform(-50, 50, size=(20, vocab_size)).astype(np.float32)

    return logits


def main() -> None:
    """Generate and save fixture files."""
    output_dir = Path(__file__).parent

    print("Generating normal logits fixture...")
    normal = generate_normal_logits()
    normal_path = output_dir / "normal_logits.npy"
    np.save(normal_path, normal)
    print(f"  Saved {normal_path} -- shape={normal.shape}, dtype={normal.dtype}")

    print("Generating corrupted logits fixture...")
    corrupted = generate_corrupted_logits()
    corrupted_path = output_dir / "corrupted_logits.npy"
    np.save(corrupted_path, corrupted)
    print(f"  Saved {corrupted_path} -- shape={corrupted.shape}, dtype={corrupted.dtype}")

    # Sanity check
    print("\nSanity checks:")
    print(f"  Normal mean: {np.mean(normal):.4f}, std: {np.std(normal):.4f}")
    print(f"  Corrupted mean: {np.mean(corrupted):.4f}, std: {np.std(corrupted):.4f}")
    print(f"  Normal range: [{np.min(normal):.2f}, {np.max(normal):.2f}]")
    print(f"  Corrupted range: [{np.min(corrupted):.2f}, {np.max(corrupted):.2f}]")


if __name__ == "__main__":
    main()
