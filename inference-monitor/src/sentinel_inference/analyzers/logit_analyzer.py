"""EWMA control chart analyzer for logit statistics.

Tracks mean, variance, kurtosis, min, max, and entropy of output logit
vectors using Exponentially Weighted Moving Average (EWMA) control charts.
EWMA is more sensitive to small sustained shifts than Shewhart charts,
making it well-suited for detecting gradual SDC degradation.

Control chart formulas:
    EWMA_t = lambda * S_t + (1 - lambda) * EWMA_{t-1}
    sigma_ewma = sigma * sqrt(lambda / (2 - lambda))
    UCL = target + L * sigma_ewma
    LCL = target - L * sigma_ewma
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import structlog

from sentinel_inference.config import LogitAnalyzerConfig

logger = structlog.get_logger(__name__)


@dataclasses.dataclass
class AnomalyEvent:
    """Represents a detected anomaly from the logit analyzer."""

    analyzer: str
    stat_name: str
    observed_value: float
    ewma_value: float
    ucl: float
    lcl: float
    sample_count: int
    severity: str  # "warning" or "critical"
    details: dict[str, Any] = dataclasses.field(default_factory=dict)


class EWMATracker:
    """Single EWMA control chart for one statistic.

    Parameters
    ----------
    lambda_ : float
        Smoothing factor (0 < lambda_ <= 1).  Smaller values give more
        weight to history, making the chart more sensitive to small shifts.
    L : float
        Control limit multiplier in sigma units.
    burn_in : int
        Number of observations before the chart starts alerting.
    """

    __slots__ = (
        "_lambda",
        "_L",
        "_burn_in",
        "_count",
        "_ewma",
        "_target",
        "_sum",
        "_sum_sq",
        "_sigma",
    )

    def __init__(self, lambda_: float = 0.1, L: float = 3.5, burn_in: int = 1000) -> None:
        self._lambda = lambda_
        self._L = L
        self._burn_in = burn_in
        self._count: int = 0
        self._ewma: float = 0.0
        self._target: float = 0.0
        self._sum: float = 0.0
        self._sum_sq: float = 0.0
        self._sigma: float = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def ewma(self) -> float:
        return self._ewma

    @property
    def target(self) -> float:
        return self._target

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def ucl(self) -> float:
        """Upper Control Limit."""
        sigma_ewma = self._sigma * np.sqrt(self._lambda / (2.0 - self._lambda))
        return self._target + self._L * sigma_ewma

    @property
    def lcl(self) -> float:
        """Lower Control Limit."""
        sigma_ewma = self._sigma * np.sqrt(self._lambda / (2.0 - self._lambda))
        return self._target - self._L * sigma_ewma

    @property
    def in_burn_in(self) -> bool:
        return self._count < self._burn_in

    def update(self, value: float) -> bool:
        """Update the tracker with a new observation.

        Returns True if the value falls outside control limits (anomaly)
        AFTER the burn-in period.
        """
        self._count += 1
        self._sum += value
        self._sum_sq += value * value

        if self._count == 1:
            self._ewma = value
            self._target = value
            self._sigma = 0.0
            return False

        # Welford-style running mean and variance for target & sigma
        self._target = self._sum / self._count
        variance = (self._sum_sq / self._count) - (self._target ** 2)
        self._sigma = np.sqrt(max(variance, 1e-15))

        # EWMA update
        self._ewma = self._lambda * value + (1.0 - self._lambda) * self._ewma

        # Check control limits only after burn-in
        if self._count < self._burn_in:
            return False

        if self._sigma < 1e-12:
            return False

        return bool(self._ewma > self.ucl or self._ewma < self.lcl)

    def reset(self) -> None:
        """Reset the tracker to initial state."""
        self._count = 0
        self._ewma = 0.0
        self._target = 0.0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._sigma = 0.0


def _compute_logit_stats(logits: np.ndarray) -> dict[str, float]:
    """Compute summary statistics of a logit vector.

    Parameters
    ----------
    logits : np.ndarray
        1-D or 2-D array of logit values.  If 2-D, statistics are
        computed over the flattened last axis (vocab dimension).

    Returns
    -------
    dict mapping stat name to float value.
    """
    if logits.ndim > 1:
        logits = logits.reshape(-1)

    logits = logits.astype(np.float64)

    mean = float(np.mean(logits))
    var = float(np.var(logits))
    min_val = float(np.min(logits))
    max_val = float(np.max(logits))

    # Kurtosis (excess kurtosis, Fisher definition)
    if var > 1e-15:
        centered = logits - mean
        m4 = float(np.mean(centered**4))
        kurtosis = m4 / (var**2) - 3.0
    else:
        kurtosis = 0.0

    # Shannon entropy of softmax distribution
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits)
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-30, None)
    entropy = float(-np.sum(probs * np.log(probs)))

    return {
        "mean": mean,
        "variance": var,
        "kurtosis": kurtosis,
        "min": min_val,
        "max": max_val,
        "entropy": entropy,
    }


class LogitAnalyzer:
    """EWMA control chart analyzer for logit output statistics.

    Maintains one EWMA tracker per tracked statistic and emits anomaly
    events when any statistic breaches its control limits.

    Parameters
    ----------
    config : LogitAnalyzerConfig
        Configuration controlling which statistics to track and EWMA
        parameters.
    """

    def __init__(self, config: LogitAnalyzerConfig | None = None) -> None:
        self._config = config or LogitAnalyzerConfig()
        self._trackers: dict[str, EWMATracker] = {}
        for stat in self._config.tracked_stats:
            self._trackers[stat] = EWMATracker(
                lambda_=self._config.ewma.lambda_,
                L=self._config.ewma.L,
                burn_in=self._config.ewma.burn_in,
            )

    @property
    def trackers(self) -> dict[str, EWMATracker]:
        return self._trackers

    def analyze(self, logits: np.ndarray) -> list[AnomalyEvent]:
        """Analyze a logit tensor and return any anomaly events.

        Parameters
        ----------
        logits : np.ndarray
            Output logit tensor from the inference server.

        Returns
        -------
        list of AnomalyEvent for any statistics that breached control limits.
        """
        if not self._config.enabled:
            return []

        stats = _compute_logit_stats(logits)
        anomalies: list[AnomalyEvent] = []

        for stat_name, tracker in self._trackers.items():
            if stat_name not in stats:
                continue
            value = stats[stat_name]
            is_anomaly = tracker.update(value)

            if is_anomaly:
                # Determine severity based on how far outside limits
                sigma_ewma = tracker.sigma * np.sqrt(
                    self._config.ewma.lambda_ / (2.0 - self._config.ewma.lambda_)
                )
                if sigma_ewma > 1e-12:
                    deviation = abs(tracker.ewma - tracker.target) / sigma_ewma
                    severity = "critical" if deviation > self._config.ewma.L * 1.5 else "warning"
                else:
                    severity = "warning"

                event = AnomalyEvent(
                    analyzer="logit_analyzer",
                    stat_name=stat_name,
                    observed_value=value,
                    ewma_value=tracker.ewma,
                    ucl=tracker.ucl,
                    lcl=tracker.lcl,
                    sample_count=tracker.count,
                    severity=severity,
                    details={
                        "target": tracker.target,
                        "sigma": tracker.sigma,
                    },
                )
                anomalies.append(event)
                logger.warning(
                    "logit_anomaly_detected",
                    stat=stat_name,
                    ewma=tracker.ewma,
                    ucl=tracker.ucl,
                    lcl=tracker.lcl,
                    severity=severity,
                )

        return anomalies

    def reset(self) -> None:
        """Reset all trackers."""
        for tracker in self._trackers.values():
            tracker.reset()
