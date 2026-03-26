"""Cross-replica KL divergence detector.

Matches inference requests across replicas by input hash and computes the
Jensen-Shannon Divergence (symmetric KL) of their softmax output
distributions.  Normal floating-point variation yields JSD < 0.001 nats;
SDC corruption causes divergence >> 0.01 nats.
"""

from __future__ import annotations

import dataclasses
import time
from collections import OrderedDict

import numpy as np
import structlog

from sentinel_inference.analyzers.logit_analyzer import AnomalyEvent
from sentinel_inference.config import KLDivergenceConfig

logger = structlog.get_logger(__name__)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x = logits.astype(np.float64)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(P || Q) in nats.

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions (must sum to 1 along last axis).

    Returns
    -------
    float
        KL divergence in nats.  Always >= 0.
    """
    p = np.clip(p.astype(np.float64), 1e-30, None)
    q = np.clip(q.astype(np.float64), 1e-30, None)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the Jensen-Shannon Divergence: JSD(P || Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M).

    Parameters
    ----------
    p, q : np.ndarray
        Probability distributions.

    Returns
    -------
    float
        JSD in nats.  Always in [0, ln(2)].
    """
    p = np.clip(p.astype(np.float64), 1e-30, None)
    q = np.clip(q.astype(np.float64), 1e-30, None)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


@dataclasses.dataclass(frozen=True, slots=True)
class PendingOutput:
    """An output distribution waiting for its cross-replica match."""

    probs: np.ndarray
    replica_id: str
    request_id: str
    timestamp: float


class KLDivergenceDetector:
    """Detect cross-replica divergence via Jensen-Shannon Divergence.

    When two replicas process the same input (matched by input hash), their
    softmax outputs should be nearly identical.  This detector maintains a
    time-windowed buffer of pending outputs and checks for matches.

    Parameters
    ----------
    config : KLDivergenceConfig
        Detection thresholds and matching parameters.
    replica_id : str
        This replica's identifier.
    """

    def __init__(
        self,
        config: KLDivergenceConfig | None = None,
        replica_id: str = "local",
    ) -> None:
        self._config = config or KLDivergenceConfig()
        self._replica_id = replica_id
        # input_hash -> list of PendingOutput
        self._pending: OrderedDict[str, list[PendingOutput]] = OrderedDict()
        self._max_pending = 10000

    def submit(
        self,
        logits: np.ndarray,
        input_hash: str,
        request_id: str,
        replica_id: str | None = None,
    ) -> list[AnomalyEvent]:
        """Submit an output for cross-replica matching.

        If a matching output from a different replica already exists in the
        buffer, compute JSD and return any anomaly events.

        Parameters
        ----------
        logits : np.ndarray
            Raw output logits.
        input_hash : str
            SHA-256 of the input for matching.
        request_id : str
            Request identifier.
        replica_id : str or None
            Source replica.  Defaults to ``self._replica_id``.

        Returns
        -------
        list of AnomalyEvent.
        """
        if not self._config.enabled:
            return []

        rid = replica_id or self._replica_id
        now = time.monotonic()

        # Evict expired entries
        self._evict_expired(now)

        probs = softmax(logits.reshape(-1))
        new_entry = PendingOutput(
            probs=probs,
            replica_id=rid,
            request_id=request_id,
            timestamp=now,
        )

        anomalies: list[AnomalyEvent] = []

        if input_hash in self._pending:
            existing = self._pending[input_hash]
            for entry in existing:
                if entry.replica_id == rid:
                    continue  # Same replica, skip

                # Found a cross-replica match
                if self._config.use_jsd:
                    divergence = jensen_shannon_divergence(probs, entry.probs)
                else:
                    divergence = kl_divergence(probs, entry.probs)

                if divergence > self._config.threshold_nats:
                    severity = (
                        "critical"
                        if divergence > self._config.threshold_nats * 10
                        else "warning"
                    )
                    anomalies.append(
                        AnomalyEvent(
                            analyzer="kl_divergence",
                            stat_name="jsd" if self._config.use_jsd else "kl",
                            observed_value=divergence,
                            ewma_value=divergence,
                            ucl=self._config.threshold_nats,
                            lcl=0.0,
                            sample_count=0,
                            severity=severity,
                            details={
                                "input_hash": input_hash,
                                "replica_a": entry.replica_id,
                                "replica_b": rid,
                                "request_a": entry.request_id,
                                "request_b": request_id,
                            },
                        )
                    )
                    logger.warning(
                        "cross_replica_divergence",
                        divergence=divergence,
                        threshold=self._config.threshold_nats,
                        replica_a=entry.replica_id,
                        replica_b=rid,
                    )

            # Add to existing list
            existing.append(new_entry)
        else:
            self._pending[input_hash] = [new_entry]

        # Enforce max size
        while len(self._pending) > self._max_pending:
            self._pending.popitem(last=False)

        return anomalies

    def _evict_expired(self, now: float) -> None:
        """Remove entries older than the match window."""
        window = self._config.match_window_seconds
        keys_to_remove: list[str] = []
        for key, entries in self._pending.items():
            entries[:] = [e for e in entries if (now - e.timestamp) < window]
            if not entries:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._pending[key]

    def reset(self) -> None:
        self._pending.clear()
