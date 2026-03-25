"""Recomputation engine for SDC verification.

When a GPU is suspected of producing corrupted results, this module replays
recent training steps on a different GPU and compares the outputs to confirm
the corruption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np
import structlog

from sentinel_training.common.config import RecomputationConfig

logger = structlog.get_logger(__name__)


class RecomputationStatus(str, Enum):
    """Status of a recomputation verification."""

    PENDING = "pending"
    RUNNING = "running"
    CONFIRMED_SDC = "confirmed_sdc"
    FALSE_ALARM = "false_alarm"
    FAILED = "failed"
    TIMEOUT = "timeout"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass
class RecomputationResult:
    """Result of a recomputation verification run."""

    status: RecomputationStatus
    suspect_rank: int
    target_device: str
    steps_replayed: int
    max_divergence: float
    divergence_threshold: float
    duration_seconds: float
    step_divergences: list[float] = field(default_factory=list)
    error_message: str | None = None

    @property
    def is_sdc_confirmed(self) -> bool:
        """Whether SDC was confirmed by recomputation."""
        return self.status == RecomputationStatus.CONFIRMED_SDC

    def to_dict(self) -> dict[str, object]:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "suspect_rank": self.suspect_rank,
            "target_device": self.target_device,
            "steps_replayed": self.steps_replayed,
            "max_divergence": self.max_divergence,
            "divergence_threshold": self.divergence_threshold,
            "duration_seconds": self.duration_seconds,
            "step_divergences": self.step_divergences,
            "error_message": self.error_message,
        }


class CheckpointLoader(Protocol):
    """Protocol for loading a model checkpoint."""

    def load(self, path: str, device: str) -> dict[str, Any]: ...


class StepReplayer(Protocol):
    """Protocol for replaying a training step."""

    def replay_step(
        self,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        batch: Any,
        device: str,
    ) -> dict[str, Any]: ...


class RecomputationEngine:
    """Engine for verifying SDC by replaying training steps on a different device.

    When a suspect GPU is flagged, this engine:
    1. Loads the last known-good checkpoint onto a different device
    2. Replays N recent training steps
    3. Compares output parameter states
    4. Reports whether divergence confirms SDC

    Args:
        config: Recomputation configuration.
        checkpoint_loader: Callable to load checkpoint state dicts.
        step_replayer: Callable to replay individual training steps.
    """

    def __init__(
        self,
        config: RecomputationConfig,
        checkpoint_loader: CheckpointLoader | None = None,
        step_replayer: StepReplayer | None = None,
    ) -> None:
        self._config = config
        self._checkpoint_loader = checkpoint_loader
        self._step_replayer = step_replayer
        self._pending_verifications: list[dict[str, Any]] = []
        self._results: list[RecomputationResult] = []
        self._approval_callback: Callable[[int], bool] | None = None

    def set_approval_callback(self, callback: Callable[[int], bool]) -> None:
        """Set a callback for human approval of auto-rollback.

        Args:
            callback: Function that takes suspect rank and returns True to approve.
        """
        self._approval_callback = callback

    def request_verification(
        self,
        suspect_rank: int,
        checkpoint_path: str,
        batches: list[Any],
        model_state: dict[str, Any] | None = None,
        optimizer_state: dict[str, Any] | None = None,
    ) -> RecomputationResult:
        """Request recomputation verification for a suspect rank.

        Args:
            suspect_rank: The rank suspected of SDC.
            checkpoint_path: Path to the last known-good checkpoint.
            batches: List of recent training batches to replay.
            model_state: Current model state from the suspect rank (for comparison).
            optimizer_state: Current optimizer state from the suspect rank.

        Returns:
            RecomputationResult with the verification outcome.
        """
        if not self._config.enabled:
            logger.info("recomputation_disabled", suspect_rank=suspect_rank)
            return RecomputationResult(
                status=RecomputationStatus.FAILED,
                suspect_rank=suspect_rank,
                target_device="none",
                steps_replayed=0,
                max_divergence=0.0,
                divergence_threshold=self._config.divergence_threshold,
                duration_seconds=0.0,
                error_message="Recomputation is disabled",
            )

        # Check auto-rollback approval if needed
        if not self._config.auto_rollback:
            if self._approval_callback is not None:
                approved = self._approval_callback(suspect_rank)
                if not approved:
                    logger.info(
                        "recomputation_not_approved", suspect_rank=suspect_rank
                    )
                    return RecomputationResult(
                        status=RecomputationStatus.AWAITING_APPROVAL,
                        suspect_rank=suspect_rank,
                        target_device="none",
                        steps_replayed=0,
                        max_divergence=0.0,
                        divergence_threshold=self._config.divergence_threshold,
                        duration_seconds=0.0,
                    )
            else:
                logger.warning(
                    "recomputation_awaiting_approval",
                    suspect_rank=suspect_rank,
                    message="auto_rollback=false and no approval callback set",
                )
                return RecomputationResult(
                    status=RecomputationStatus.AWAITING_APPROVAL,
                    suspect_rank=suspect_rank,
                    target_device="none",
                    steps_replayed=0,
                    max_divergence=0.0,
                    divergence_threshold=self._config.divergence_threshold,
                    duration_seconds=0.0,
                )

        target_device = self._config.target_device or "cpu"
        start_time = time.monotonic()

        try:
            return self._run_verification(
                suspect_rank=suspect_rank,
                checkpoint_path=checkpoint_path,
                batches=batches,
                model_state=model_state,
                target_device=target_device,
                start_time=start_time,
            )
        except Exception as exc:
            duration = time.monotonic() - start_time
            logger.error(
                "recomputation_failed",
                suspect_rank=suspect_rank,
                error=str(exc),
            )
            result = RecomputationResult(
                status=RecomputationStatus.FAILED,
                suspect_rank=suspect_rank,
                target_device=target_device,
                steps_replayed=0,
                max_divergence=0.0,
                divergence_threshold=self._config.divergence_threshold,
                duration_seconds=duration,
                error_message=str(exc),
            )
            self._results.append(result)
            return result

    def _run_verification(
        self,
        suspect_rank: int,
        checkpoint_path: str,
        batches: list[Any],
        model_state: dict[str, Any] | None,
        target_device: str,
        start_time: float,
    ) -> RecomputationResult:
        """Execute the recomputation verification.

        Loads checkpoint, replays steps, compares states.
        """
        if self._checkpoint_loader is None or self._step_replayer is None:
            raise RuntimeError(
                "checkpoint_loader and step_replayer must be set for verification"
            )

        logger.info(
            "recomputation_start",
            suspect_rank=suspect_rank,
            target_device=target_device,
            replay_steps=self._config.replay_steps,
        )

        # Load the checkpoint onto the target device
        checkpoint = self._checkpoint_loader.load(checkpoint_path, target_device)
        replay_model_state = checkpoint.get("model_state", {})
        replay_optimizer_state = checkpoint.get("optimizer_state", {})

        # Replay steps
        steps_to_replay = min(self._config.replay_steps, len(batches))
        step_divergences: list[float] = []

        for step_idx in range(steps_to_replay):
            elapsed = time.monotonic() - start_time
            if elapsed > self._config.timeout_seconds:
                logger.warning(
                    "recomputation_timeout",
                    steps_completed=step_idx,
                    elapsed=elapsed,
                )
                result = RecomputationResult(
                    status=RecomputationStatus.TIMEOUT,
                    suspect_rank=suspect_rank,
                    target_device=target_device,
                    steps_replayed=step_idx,
                    max_divergence=max(step_divergences) if step_divergences else 0.0,
                    divergence_threshold=self._config.divergence_threshold,
                    duration_seconds=elapsed,
                    step_divergences=step_divergences,
                )
                self._results.append(result)
                return result

            new_state = self._step_replayer.replay_step(
                model_state=replay_model_state,
                optimizer_state=replay_optimizer_state,
                batch=batches[step_idx],
                device=target_device,
            )
            replay_model_state = new_state.get("model_state", replay_model_state)
            replay_optimizer_state = new_state.get(
                "optimizer_state", replay_optimizer_state
            )

        # Compute divergence between suspect state and replayed state
        if model_state is not None:
            max_div = self._compute_state_divergence(model_state, replay_model_state)
            step_divergences.append(max_div)
        else:
            max_div = 0.0

        duration = time.monotonic() - start_time

        if max_div > self._config.divergence_threshold:
            status = RecomputationStatus.CONFIRMED_SDC
            logger.critical(
                "sdc_confirmed",
                suspect_rank=suspect_rank,
                max_divergence=max_div,
                threshold=self._config.divergence_threshold,
            )
        else:
            status = RecomputationStatus.FALSE_ALARM
            logger.info(
                "sdc_false_alarm",
                suspect_rank=suspect_rank,
                max_divergence=max_div,
            )

        result = RecomputationResult(
            status=status,
            suspect_rank=suspect_rank,
            target_device=target_device,
            steps_replayed=steps_to_replay,
            max_divergence=max_div,
            divergence_threshold=self._config.divergence_threshold,
            duration_seconds=duration,
            step_divergences=step_divergences,
        )
        self._results.append(result)
        return result

    @staticmethod
    def _compute_state_divergence(
        state_a: dict[str, Any],
        state_b: dict[str, Any],
    ) -> float:
        """Compute the maximum parameter-wise divergence between two state dicts.

        Uses the relative L2 norm of the difference for each parameter.

        Args:
            state_a: First state dictionary.
            state_b: Second state dictionary.

        Returns:
            Maximum relative divergence across all shared parameters.
        """
        max_divergence = 0.0
        common_keys = set(state_a.keys()) & set(state_b.keys())

        for key in common_keys:
            a = state_a[key]
            b = state_b[key]
            try:
                a_np = np.asarray(a, dtype=np.float64).ravel()
                b_np = np.asarray(b, dtype=np.float64).ravel()
                if a_np.shape != b_np.shape:
                    continue
                diff_norm = float(np.linalg.norm(a_np - b_np))
                base_norm = float(np.linalg.norm(a_np))
                if base_norm > 1e-12:
                    rel_div = diff_norm / base_norm
                else:
                    rel_div = diff_norm
                max_divergence = max(max_divergence, rel_div)
            except (TypeError, ValueError):
                continue

        return max_divergence

    @property
    def results(self) -> list[RecomputationResult]:
        """All verification results."""
        return list(self._results)

    def cost_estimate(self, step_time_seconds: float) -> float:
        """Estimate the wall-clock cost of a recomputation verification.

        Args:
            step_time_seconds: Average time per training step.

        Returns:
            Estimated total verification time in seconds.
        """
        return self._config.replay_steps * step_time_seconds
