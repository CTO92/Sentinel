"""Tests for the KL divergence cross-replica detector."""

from __future__ import annotations

import numpy as np

from sentinel_inference.analyzers.kl_divergence import (
    KLDivergenceDetector,
    jensen_shannon_divergence,
    kl_divergence,
    softmax,
)
from sentinel_inference.config import KLDivergenceConfig


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        logits = np.array([1.0, 2.0, 3.0])
        probs = softmax(logits)
        assert abs(np.sum(probs) - 1.0) < 1e-10

    def test_numerical_stability(self) -> None:
        """Softmax should handle large logits without overflow."""
        logits = np.array([1000.0, 1001.0, 1002.0])
        probs = softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(np.sum(probs) - 1.0) < 1e-10

    def test_negative_logits(self) -> None:
        logits = np.array([-1000.0, -999.0, -998.0])
        probs = softmax(logits)
        assert np.all(np.isfinite(probs))
        assert abs(np.sum(probs) - 1.0) < 1e-10


class TestKLDivergence:
    def test_identical_distributions(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(kl_divergence(p, p)) < 1e-10

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(100):
            p = rng.dirichlet(np.ones(10))
            q = rng.dirichlet(np.ones(10))
            assert kl_divergence(p, q) >= -1e-10

    def test_asymmetric(self) -> None:
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        # KL is asymmetric in general
        assert kl_pq > 0
        assert kl_qp > 0
        # But for this symmetric case they happen to be equal
        assert abs(kl_pq - kl_qp) < 1e-10


class TestJensenShannonDivergence:
    def test_identical_distributions(self) -> None:
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(jensen_shannon_divergence(p, p)) < 1e-10

    def test_symmetric(self) -> None:
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        assert abs(jensen_shannon_divergence(p, q) - jensen_shannon_divergence(q, p)) < 1e-10

    def test_bounded(self) -> None:
        """JSD should be in [0, ln(2)]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            p = rng.dirichlet(np.ones(10))
            q = rng.dirichlet(np.ones(10))
            jsd = jensen_shannon_divergence(p, q)
            assert -1e-10 <= jsd <= np.log(2) + 1e-10

    def test_maximum_divergence(self) -> None:
        """Disjoint distributions should have JSD close to ln(2)."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        jsd = jensen_shannon_divergence(p, q)
        assert abs(jsd - np.log(2)) < 0.01


class TestKLDivergenceDetector:
    def test_no_anomaly_same_output(self) -> None:
        """Same logits from different replicas should not trigger anomaly."""
        config = KLDivergenceConfig(threshold_nats=0.01)
        detector = KLDivergenceDetector(config=config, replica_id="replica_a")

        logits = np.random.randn(1000).astype(np.float32)
        input_hash = "test_hash_1"

        # Submit from replica A
        events_a = detector.submit(logits, input_hash, "req1", replica_id="replica_a")
        assert len(events_a) == 0

        # Submit same logits from replica B
        events_b = detector.submit(logits, input_hash, "req2", replica_id="replica_b")
        assert len(events_b) == 0

    def test_detects_divergent_outputs(self) -> None:
        """Very different logits from different replicas should trigger anomaly."""
        config = KLDivergenceConfig(threshold_nats=0.01)
        detector = KLDivergenceDetector(config=config, replica_id="replica_a")

        rng = np.random.default_rng(42)
        logits_a = rng.normal(0, 1, size=1000).astype(np.float32)
        logits_b = rng.normal(5, 1, size=1000).astype(np.float32)
        input_hash = "test_hash_2"

        detector.submit(logits_a, input_hash, "req1", replica_id="replica_a")
        events = detector.submit(logits_b, input_hash, "req2", replica_id="replica_b")
        assert len(events) > 0
        assert events[0].analyzer == "kl_divergence"

    def test_same_replica_ignored(self) -> None:
        """Outputs from the same replica should not be compared."""
        config = KLDivergenceConfig(threshold_nats=0.01)
        detector = KLDivergenceDetector(config=config, replica_id="replica_a")

        rng = np.random.default_rng(42)
        logits_a = rng.normal(0, 1, size=100).astype(np.float32)
        logits_b = rng.normal(10, 1, size=100).astype(np.float32)
        input_hash = "test_hash_3"

        detector.submit(logits_a, input_hash, "req1", replica_id="replica_a")
        events = detector.submit(logits_b, input_hash, "req2", replica_id="replica_a")
        assert len(events) == 0

    def test_different_input_hash_not_matched(self) -> None:
        config = KLDivergenceConfig(threshold_nats=0.01)
        detector = KLDivergenceDetector(config=config)

        rng = np.random.default_rng(42)
        logits_a = rng.normal(0, 1, size=100).astype(np.float32)
        logits_b = rng.normal(10, 1, size=100).astype(np.float32)

        detector.submit(logits_a, "hash_a", "req1", replica_id="r1")
        events = detector.submit(logits_b, "hash_b", "req2", replica_id="r2")
        assert len(events) == 0

    def test_disabled(self) -> None:
        config = KLDivergenceConfig(enabled=False)
        detector = KLDivergenceDetector(config=config)
        logits = np.random.randn(100).astype(np.float32)
        assert detector.submit(logits, "h", "r") == []

    def test_severity_levels(self) -> None:
        """Large divergence should produce critical severity."""
        config = KLDivergenceConfig(threshold_nats=0.001)
        detector = KLDivergenceDetector(config=config)

        # Create extremely divergent distributions
        logits_a = np.zeros(100, dtype=np.float32)
        logits_a[0] = 100.0
        logits_b = np.zeros(100, dtype=np.float32)
        logits_b[99] = 100.0

        detector.submit(logits_a, "h1", "r1", replica_id="r1")
        events = detector.submit(logits_b, "h1", "r2", replica_id="r2")
        assert len(events) > 0
        # Very high divergence should be critical
        assert events[0].severity == "critical"
