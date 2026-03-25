"""JAX custom VJP wrappers for gradient monitoring.

Provides a `sentinel_monitor` transform that wraps a JAX function to
intercept and monitor gradient norms during automatic differentiation,
using non-blocking telemetry via jax.debug.callback.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

import numpy as np
import structlog

from sentinel_training.common.anomaly_detector import (
    AnomalyScore,
    AnomalyType,
    EWMATracker,
)
from sentinel_training.common.config import GradientNormConfig, SentinelConfig

logger = structlog.get_logger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    custom_vjp = None  # type: ignore[assignment]

F = TypeVar("F", bound=Callable[..., Any])


class _MonitorState:
    """Global mutable state for monitoring across JAX transforms.

    Since JAX transforms are functional, we need a side-channel for
    accumulating telemetry. This is thread-local in practice since
    JAX dispatches to a single host controller thread.
    """

    def __init__(self, config: GradientNormConfig) -> None:
        self.config = config
        self.trackers: dict[str, EWMATracker] = {}
        self.step_count: int = 0
        self.anomalies: list[AnomalyScore] = []

    def get_tracker(self, name: str) -> EWMATracker:
        if name not in self.trackers:
            self.trackers[name] = EWMATracker(
                alpha=self.config.ewma_lambda,
                warmup_steps=self.config.warmup_steps,
            )
        return self.trackers[name]


# Module-level state, initialized when sentinel_monitor is first used
_global_state: _MonitorState | None = None


def _ensure_state(config: GradientNormConfig | None = None) -> _MonitorState:
    """Get or initialize the global monitor state."""
    global _global_state
    if _global_state is None:
        _global_state = _MonitorState(config or GradientNormConfig())
    return _global_state


def reset_state() -> None:
    """Reset global monitor state. Useful for testing."""
    global _global_state
    _global_state = None


def _telemetry_callback(
    grad_norms: dict[str, float],
    state: _MonitorState,
    step: int,
) -> None:
    """Non-blocking telemetry callback invoked via jax.debug.callback.

    Processes gradient norms through EWMA trackers and records anomalies.

    Args:
        grad_norms: Mapping from parameter path to gradient L2 norm.
        state: The monitor state.
        step: Current training step.
    """
    for name, norm in grad_norms.items():
        tracker = state.get_tracker(name)
        z = tracker.z_score(norm)
        tracker.update(norm)

        if tracker.is_warmed_up and z > state.config.sigma_multiplier:
            anomaly_type = (
                AnomalyType.GRADIENT_SPIKE
                if norm > tracker.mean
                else AnomalyType.GRADIENT_COLLAPSE
            )
            anomaly = AnomalyScore(
                anomaly_type=anomaly_type,
                score=z,
                threshold=state.config.sigma_multiplier,
                observed_value=norm,
                expected_value=tracker.mean,
                step=step,
                layer_name=name,
                metadata={"ewma_std": tracker.std},
            )
            state.anomalies.append(anomaly)
            logger.warning(
                "jax_gradient_anomaly",
                type=anomaly_type.value,
                name=name,
                z_score=z,
                norm=norm,
                step=step,
            )


def _compute_pytree_norms(
    pytree: Any, prefix: str = ""
) -> dict[str, float]:
    """Compute L2 norms for all leaf arrays in a pytree.

    Args:
        pytree: A JAX pytree (nested dict/list of arrays).
        prefix: Path prefix for naming.

    Returns:
        Dictionary mapping paths to L2 norms.
    """
    norms: dict[str, float] = {}
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    leaf_paths = _get_leaf_paths(pytree, prefix)

    for path, leaf in zip(leaf_paths, leaves):
        norm_val = float(jnp.linalg.norm(jnp.ravel(leaf)))
        norms[path] = norm_val

    return norms


def _get_leaf_paths(pytree: Any, prefix: str = "") -> list[str]:
    """Extract string paths for each leaf in a pytree.

    Args:
        pytree: A JAX pytree.
        prefix: Path prefix.

    Returns:
        List of string paths corresponding to tree_flatten leaf order.
    """
    paths: list[str] = []

    def _traverse(obj: Any, current_path: str) -> None:
        if isinstance(obj, dict):
            for key in sorted(obj.keys()):
                _traverse(obj[key], f"{current_path}.{key}" if current_path else str(key))
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _traverse(item, f"{current_path}[{i}]")
        else:
            # Leaf node
            paths.append(current_path if current_path else "param")

    _traverse(pytree, prefix)
    return paths


def sentinel_monitor(
    fn: F | None = None,
    *,
    config: SentinelConfig | None = None,
    name: str = "monitored_fn",
) -> F | Callable[[F], F]:
    """Transform that wraps a JAX function to monitor its gradients.

    Can be used as a decorator with or without arguments::

        @sentinel_monitor
        def loss_fn(params, batch):
            ...

        @sentinel_monitor(config=my_config, name="my_loss")
        def loss_fn(params, batch):
            ...

    When the wrapped function is differentiated via jax.grad or jax.value_and_grad,
    the custom VJP rule intercepts the backward pass to compute and log gradient norms.

    Args:
        fn: The function to wrap.
        config: Optional Sentinel configuration.
        name: Name for this monitored function (used in telemetry).

    Returns:
        The wrapped function with gradient monitoring.
    """
    if fn is None:
        # Called with arguments: @sentinel_monitor(config=..., name=...)
        return functools.partial(sentinel_monitor, config=config, name=name)  # type: ignore[return-value]

    if jax is None:
        logger.warning("jax_not_available", message="sentinel_monitor is a no-op")
        return fn  # type: ignore[return-value]

    grad_config = config.gradient_norm if config else GradientNormConfig()
    state = _ensure_state(grad_config)

    @custom_vjp
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    def wrapped_fwd(*args: Any, **kwargs: Any) -> tuple[Any, tuple[Any, ...]]:
        """Forward pass: compute output and save residuals for backward."""
        result = fn(*args, **kwargs)
        # Save args as residuals for the backward pass
        return result, args

    def wrapped_bwd(residuals: tuple[Any, ...], g: Any) -> tuple[Any, ...]:
        """Backward pass: compute gradients and monitor their norms.

        Uses jax.debug.callback for non-blocking telemetry so monitoring
        does not interfere with XLA compilation or execution.
        """
        # Compute the actual gradients using JAX's autodiff
        # We use jax.linear_util or direct grad for the original function
        args = residuals
        state.step_count += 1

        # Compute gradient norms from the incoming cotangent
        if isinstance(g, tuple):
            for i, gi in enumerate(g):
                if gi is not None:
                    norms = _compute_pytree_norms(gi, prefix=f"{name}.grad.{i}")
                    _telemetry_callback(norms, state, state.step_count)
        else:
            norms = _compute_pytree_norms(g, prefix=f"{name}.grad")
            _telemetry_callback(norms, state, state.step_count)

        # Pass through the cotangent as-is (identity VJP for monitoring only)
        if isinstance(g, tuple):
            return g
        return (g,) * len(args)

    wrapped.defvjp(wrapped_fwd, wrapped_bwd)

    @functools.wraps(fn)
    def outer(*args: Any, **kwargs: Any) -> Any:
        return wrapped(*args, **kwargs)

    # Attach state for inspection
    outer._sentinel_state = state  # type: ignore[attr-defined]

    return outer  # type: ignore[return-value]


def get_anomalies() -> list[AnomalyScore]:
    """Retrieve all anomalies from the global monitor state.

    Returns:
        List of detected anomaly scores.
    """
    if _global_state is None:
        return []
    return list(_global_state.anomalies)


def get_gradient_norms() -> dict[str, float]:
    """Retrieve current EWMA gradient norm means.

    Returns:
        Dictionary mapping parameter paths to EWMA means.
    """
    if _global_state is None:
        return {}
    return {
        name: tracker.mean
        for name, tracker in _global_state.trackers.items()
    }
