"""Tests for cross-rank gradient divergence detection with mocked distributed ops."""

from __future__ import annotations

import pytest

from sentinel_training.common.anomaly_detector import AnomalyType
from sentinel_training.common.config import CrossRankDivergenceConfig

torch = pytest.importorskip("torch")

from sentinel_training.pytorch.ddp_divergence import DDPDivergenceDetector  # noqa: E402


class TestDDPDivergenceDetector:
    """Tests for the DDP gradient divergence detector."""

    def _make_config(self, **kwargs: object) -> CrossRankDivergenceConfig:
        defaults = {
            "check_interval": 1,  # Check every step for testing
            "hash_seed": 42,
            "projection_dim": 16,
            "tolerance": 1e-6,
        }
        defaults.update(kwargs)
        return CrossRankDivergenceConfig(**defaults)  # type: ignore[arg-type]

    def _make_model(self, seed: int = 0) -> torch.nn.Module:
        """Create a small model with deterministic gradients."""
        torch.manual_seed(seed)
        model = torch.nn.Linear(10, 5, bias=True)
        x = torch.randn(3, 10)
        y = model(x)
        y.sum().backward()
        return model

    def test_hash_computation_deterministic(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=2)

        model = self._make_model(seed=42)
        hash1 = detector.compute_gradient_hash(model)
        hash2 = detector.compute_gradient_hash(model)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest length

    def test_different_gradients_different_hash(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=2)

        model1 = self._make_model(seed=42)
        model2 = self._make_model(seed=99)

        hash1 = detector.compute_gradient_hash(model1)
        hash2 = detector.compute_gradient_hash(model2)

        assert hash1 != hash2

    def test_no_divergence_detected(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=4)

        same_hash = "a" * 64
        all_hashes = [same_hash, same_hash, same_hash, same_hash]

        result = detector.check_divergence_from_hashes(same_hash, all_hashes)
        assert result is None

    def test_single_rank_divergence(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=4)

        good_hash = "a" * 64
        bad_hash = "b" * 64
        all_hashes = [good_hash, good_hash, bad_hash, good_hash]

        result = detector.check_divergence_from_hashes(good_hash, all_hashes)

        assert result is not None
        assert result.anomaly_type == AnomalyType.CROSS_RANK_DIVERGENCE
        assert 2 in result.metadata["suspect_ranks"]
        assert len(result.metadata["suspect_ranks"]) == 1

    def test_multiple_rank_divergence(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=4)

        hash_a = "a" * 64
        hash_b = "b" * 64
        hash_c = "c" * 64
        all_hashes = [hash_a, hash_a, hash_b, hash_c]

        result = detector.check_divergence_from_hashes(hash_a, all_hashes)

        assert result is not None
        suspects = result.metadata["suspect_ranks"]
        assert 2 in suspects
        assert 3 in suspects
        assert result.metadata["unique_hash_count"] == 3

    def test_local_rank_is_suspect(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=2, world_size=4)

        good_hash = "a" * 64
        bad_hash = "b" * 64
        all_hashes = [good_hash, good_hash, bad_hash, good_hash]

        result = detector.check_divergence_from_hashes(bad_hash, all_hashes)

        assert result is not None
        assert result.metadata["is_local_suspect"] is True

    def test_check_interval_respected(self) -> None:
        config = self._make_config(check_interval=5)
        detector = DDPDivergenceDetector(config, rank=0, world_size=2)

        model = self._make_model()
        # Steps 1-4 should not check
        for _ in range(4):
            result = detector.check_divergence(model)
            assert result is None

        # Step 5 would check but needs distributed (so returns None without dist)

    def test_single_rank_skips_check(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=1)

        model = self._make_model()
        result = detector.check_divergence(model)
        assert result is None

    def test_empty_gradients_hash(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=2)

        # Model with no gradients
        model = torch.nn.Linear(10, 5)
        hash_val = detector.compute_gradient_hash(model)
        assert len(hash_val) == 64  # Still produces a valid hash

    def test_hash_from_tensor_list(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=2)

        grads = [torch.randn(10, 5), torch.randn(5)]
        hash1 = detector.compute_gradient_hash_from_tensors(grads)
        hash2 = detector.compute_gradient_hash_from_tensors(grads)
        assert hash1 == hash2

    def test_anomalies_list(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=4)

        good_hash = "a" * 64
        bad_hash = "b" * 64
        all_hashes = [good_hash, good_hash, bad_hash, good_hash]

        detector.check_divergence_from_hashes(good_hash, all_hashes)
        assert len(detector.anomalies) == 1

    def test_reset(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=4)

        good_hash = "a" * 64
        bad_hash = "b" * 64
        detector.check_divergence_from_hashes(good_hash, [good_hash, bad_hash])

        detector.reset()
        assert detector.step_count == 0
        assert len(detector.anomalies) == 0

    def test_step_count_increments(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=4)

        assert detector.step_count == 0
        same_hash = "a" * 64
        detector.check_divergence_from_hashes(same_hash, [same_hash] * 4)
        assert detector.step_count == 1
        detector.check_divergence_from_hashes(same_hash, [same_hash] * 4)
        assert detector.step_count == 2

    def test_hash_distribution_in_metadata(self) -> None:
        config = self._make_config()
        detector = DDPDivergenceDetector(config, rank=0, world_size=4)

        hash_a = "a" * 64
        hash_b = "b" * 64
        all_hashes = [hash_a, hash_a, hash_a, hash_b]

        result = detector.check_divergence_from_hashes(hash_a, all_hashes)
        assert result is not None
        dist = result.metadata["hash_distribution"]
        assert dist[hash_a] == 3
        assert dist[hash_b] == 1
