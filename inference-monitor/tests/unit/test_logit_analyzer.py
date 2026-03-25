"""Tests for the logit EWMA control chart analyzer."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel_inference.analyzers.logit_analyzer import (
    EWMATracker,
    LogitAnalyzer,
    _compute_logit_stats,
)
from sentinel_inference.config import EWMAConfig, LogitAnalyzerConfig


class TestEWMATracker:
    """Test the core EWMA tracker logic."""

    def test_initial_state(self) -> None:
        tracker = EWMATracker(lambda_=0.1, L=3.5, burn_in=100)
        assert tracker.count == 0
        assert tracker.ewma == 0.0
        assert tracker.in_burn_in is True

    def test_single_update(self) -> None:
        tracker = EWMATracker(lambda_=0.1, L=3.5, burn_in=100)
        tracker.update(5.0)
        assert tracker.count == 1
        assert tracker.ewma == 5.0

    def test_ewma_computation(self) -> None:
        """Verify EWMA formula: EWMA_t = lambda * S_t + (1 - lambda) * EWMA_{t-1}."""
        lam = 0.2
        tracker = EWMATracker(lambda_=lam, L=3.5, burn_in=5)
        values = [10.0, 12.0, 11.0, 13.0, 10.5]
        ewma = values[0]
        for v in values[1:]:
            ewma = lam * v + (1.0 - lam) * ewma
        for v in values:
            tracker.update(v)
        assert abs(tracker.ewma - ewma) < 1e-10

    def test_burn_in_no_alert(self) -> None:
        """No anomalies should be raised during burn-in period."""
        tracker = EWMATracker(lambda_=0.1, L=3.5, burn_in=100)
        # Feed steady-state values
        for _ in range(99):
            result = tracker.update(5.0)
            assert result is False
        assert tracker.in_burn_in is True

    def test_alert_after_burn_in_with_shift(self) -> None:
        """A large shift after burn-in should trigger an alert."""
        tracker = EWMATracker(lambda_=0.1, L=3.0, burn_in=50)
        rng = np.random.default_rng(42)
        # Burn-in with stable distribution
        for _ in range(60):
            tracker.update(float(rng.normal(0.0, 1.0)))
        # Inject a large shift
        alerted = False
        for _ in range(100):
            result = tracker.update(float(rng.normal(10.0, 1.0)))
            if result:
                alerted = True
                break
        assert alerted, "Expected alert after sustained shift"

    def test_no_alert_for_stable_data(self) -> None:
        """Stable data after burn-in should not trigger alerts."""
        tracker = EWMATracker(lambda_=0.1, L=3.5, burn_in=100)
        rng = np.random.default_rng(123)
        alerts = 0
        for _ in range(2000):
            result = tracker.update(float(rng.normal(0.0, 1.0)))
            if result:
                alerts += 1
        # With L=3.5, false alarm rate should be very low
        assert alerts < 10, f"Too many false alarms: {alerts}"

    def test_reset(self) -> None:
        tracker = EWMATracker()
        for i in range(10):
            tracker.update(float(i))
        tracker.reset()
        assert tracker.count == 0
        assert tracker.ewma == 0.0

    def test_control_limits(self) -> None:
        """UCL and LCL should be symmetric around target."""
        tracker = EWMATracker(lambda_=0.1, L=3.0, burn_in=10)
        rng = np.random.default_rng(0)
        for _ in range(100):
            tracker.update(float(rng.normal(5.0, 2.0)))
        ucl = tracker.ucl
        lcl = tracker.lcl
        assert ucl > tracker.target
        assert lcl < tracker.target
        assert abs((ucl - tracker.target) - (tracker.target - lcl)) < 1e-10


class TestComputeLogitStats:
    """Test the logit statistics computation."""

    def test_basic_stats(self) -> None:
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        stats = _compute_logit_stats(logits)
        assert abs(stats["mean"] - 3.0) < 1e-6
        assert abs(stats["min"] - 1.0) < 1e-6
        assert abs(stats["max"] - 5.0) < 1e-6
        assert stats["variance"] > 0
        assert stats["entropy"] > 0

    def test_2d_input(self) -> None:
        logits = np.random.randn(10, 100).astype(np.float32)
        stats = _compute_logit_stats(logits)
        assert "mean" in stats
        assert "entropy" in stats

    def test_uniform_logits_high_entropy(self) -> None:
        """Uniform logits should have high entropy."""
        logits = np.zeros(1000, dtype=np.float32)
        stats = _compute_logit_stats(logits)
        # Uniform distribution has entropy = log(N)
        expected = np.log(1000)
        assert abs(stats["entropy"] - expected) < 0.01

    def test_peaked_logits_low_entropy(self) -> None:
        """Peaked logits should have low entropy."""
        logits = np.full(1000, -100.0, dtype=np.float32)
        logits[0] = 100.0
        stats = _compute_logit_stats(logits)
        assert stats["entropy"] < 0.01


class TestLogitAnalyzer:
    """Integration tests for the full LogitAnalyzer."""

    def test_no_anomalies_during_burn_in(self) -> None:
        config = LogitAnalyzerConfig(
            ewma=EWMAConfig(**{"lambda": 0.1, "L": 3.5, "burn_in": 100})
        )
        analyzer = LogitAnalyzer(config)
        rng = np.random.default_rng(42)
        for _ in range(99):
            logits = rng.normal(0, 1, size=1000).astype(np.float32)
            anomalies = analyzer.analyze(logits)
            assert len(anomalies) == 0

    def test_detects_corruption(self) -> None:
        """Analyzer should detect a sudden distributional shift."""
        config = LogitAnalyzerConfig(
            ewma=EWMAConfig(**{"lambda": 0.1, "L": 3.0, "burn_in": 50})
        )
        analyzer = LogitAnalyzer(config)
        rng = np.random.default_rng(42)

        # Feed normal data through burn-in
        for _ in range(60):
            logits = rng.normal(0, 1, size=1000).astype(np.float32)
            analyzer.analyze(logits)

        # Now inject corrupted data (shifted mean)
        detected = False
        for _ in range(200):
            logits = rng.normal(5.0, 1.0, size=1000).astype(np.float32)
            anomalies = analyzer.analyze(logits)
            if anomalies:
                detected = True
                break
        assert detected, "Expected anomaly detection for shifted distribution"

    def test_disabled_returns_empty(self) -> None:
        config = LogitAnalyzerConfig(enabled=False)
        analyzer = LogitAnalyzer(config)
        logits = np.random.randn(100).astype(np.float32)
        assert analyzer.analyze(logits) == []

    def test_anomaly_event_fields(self) -> None:
        config = LogitAnalyzerConfig(
            ewma=EWMAConfig(**{"lambda": 0.3, "L": 2.0, "burn_in": 20})
        )
        analyzer = LogitAnalyzer(config)
        rng = np.random.default_rng(0)

        for _ in range(25):
            analyzer.analyze(rng.normal(0, 1, size=500).astype(np.float32))

        # Inject anomaly
        for _ in range(50):
            anomalies = analyzer.analyze(
                rng.normal(20.0, 1.0, size=500).astype(np.float32)
            )
            if anomalies:
                event = anomalies[0]
                assert event.analyzer == "logit_analyzer"
                assert event.stat_name in config.tracked_stats
                assert event.severity in ("warning", "critical")
                assert event.ucl > event.lcl
                return
        pytest.fail("Expected at least one anomaly event")
