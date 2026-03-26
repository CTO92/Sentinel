"""Entropy analyzer for detecting SDC-induced output distribution anomalies.

Computes Shannon entropy of the softmax distribution derived from output
logits and tracks it with an EWMA control chart.  SDC typically manifests
as either:

- **Entropy collapse**: the model becomes degenerate, outputting a peaked
  distribution (low entropy).
- **Entropy explosion**: corrupted weights produce near-uniform outputs
  (high entropy approaching log(vocab_size)).
"""

from __future__ import annotations

import numpy as np
import structlog

from sentinel_inference.analyzers.logit_analyzer import AnomalyEvent, EWMATracker
from sentinel_inference.config import EntropyAnalyzerConfig

logger = structlog.get_logger(__name__)


def compute_entropy(logits: np.ndarray) -> float:
    """Compute Shannon entropy (in nats) of softmax(logits).

    Parameters
    ----------
    logits : np.ndarray
        Raw logit vector.  If 2-D, the last axis is treated as the
        vocabulary dimension and entropy is averaged across positions.

    Returns
    -------
    float
        Shannon entropy H = -sum(p * ln(p)).
    """
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)

    logits = logits.astype(np.float64)

    # Numerically stable softmax per row
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-30, None)
    row_entropy = -np.sum(probs * np.log(probs), axis=-1)

    return float(np.mean(row_entropy))


class EntropyAnalyzer:
    """Track output entropy with EWMA and alert on collapse or explosion.

    Parameters
    ----------
    config : EntropyAnalyzerConfig
        Thresholds and EWMA parameters.
    """

    def __init__(self, config: EntropyAnalyzerConfig | None = None) -> None:
        self._config = config or EntropyAnalyzerConfig()
        self._tracker = EWMATracker(
            lambda_=self._config.ewma.lambda_,
            L=self._config.ewma.L,
            burn_in=self._config.ewma.burn_in,
        )

    @property
    def tracker(self) -> EWMATracker:
        return self._tracker

    def analyze(self, logits: np.ndarray) -> list[AnomalyEvent]:
        """Analyze a logit tensor for entropy anomalies.

        Parameters
        ----------
        logits : np.ndarray
            Output logit tensor.

        Returns
        -------
        list of AnomalyEvent (may contain multiple if both EWMA and
        threshold violations occur).
        """
        if not self._config.enabled:
            return []

        entropy = compute_entropy(logits)
        anomalies: list[AnomalyEvent] = []

        # EWMA control chart check
        ewma_breach = self._tracker.update(entropy)
        if ewma_breach:
            anomalies.append(
                AnomalyEvent(
                    analyzer="entropy_analyzer",
                    stat_name="entropy_ewma",
                    observed_value=entropy,
                    ewma_value=self._tracker.ewma,
                    ucl=self._tracker.ucl,
                    lcl=self._tracker.lcl,
                    sample_count=self._tracker.count,
                    severity="warning",
                )
            )

        # Absolute threshold checks (always active, even during burn-in)
        if entropy < self._config.min_entropy:
            anomalies.append(
                AnomalyEvent(
                    analyzer="entropy_analyzer",
                    stat_name="entropy_collapse",
                    observed_value=entropy,
                    ewma_value=self._tracker.ewma,
                    ucl=self._config.max_entropy,
                    lcl=self._config.min_entropy,
                    sample_count=self._tracker.count,
                    severity="critical",
                    details={"threshold": self._config.min_entropy, "type": "collapse"},
                )
            )
            logger.warning(
                "entropy_collapse_detected",
                entropy=entropy,
                threshold=self._config.min_entropy,
            )

        if entropy > self._config.max_entropy:
            anomalies.append(
                AnomalyEvent(
                    analyzer="entropy_analyzer",
                    stat_name="entropy_explosion",
                    observed_value=entropy,
                    ewma_value=self._tracker.ewma,
                    ucl=self._config.max_entropy,
                    lcl=self._config.min_entropy,
                    sample_count=self._tracker.count,
                    severity="critical",
                    details={"threshold": self._config.max_entropy, "type": "explosion"},
                )
            )
            logger.warning(
                "entropy_explosion_detected",
                entropy=entropy,
                threshold=self._config.max_entropy,
            )

        return anomalies

    def reset(self) -> None:
        self._tracker.reset()
