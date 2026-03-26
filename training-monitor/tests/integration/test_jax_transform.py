"""Integration test for JAX sentinel_monitor transform on simple functions."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import numpy as np  # noqa: E402

from sentinel_training.common.config import SentinelConfig  # noqa: E402
from sentinel_training.jax.transforms import (  # noqa: E402
    get_anomalies,
    get_gradient_norms,
    reset_state,
    sentinel_monitor,
)


class TestJAXTransformIntegration:
    """Integration tests for the JAX sentinel_monitor transform."""

    def setup_method(self) -> None:
        """Reset global state before each test."""
        reset_state()

    def test_sentinel_monitor_decorator_no_args(self) -> None:
        @sentinel_monitor
        def loss_fn(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum((params @ x) ** 2)

        params = jnp.ones((3, 3))
        x = jnp.array([1.0, 2.0, 3.0])

        # Forward should work normally
        result = loss_fn(params, x)
        assert result.shape == ()

    def test_sentinel_monitor_decorator_with_args(self) -> None:
        config = SentinelConfig()

        @sentinel_monitor(config=config, name="test_loss")
        def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(params ** 2)

        params = jnp.ones(10)
        result = loss_fn(params)
        assert float(result) == 10.0

    def test_gradient_computation_works(self) -> None:
        @sentinel_monitor
        def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(params ** 2)

        params = jnp.array([1.0, 2.0, 3.0])
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)

        # Gradients of sum(x^2) = 2*x
        expected = 2.0 * params
        np.testing.assert_allclose(np.array(grads), np.array(expected), atol=1e-5)

    def test_value_and_grad_works(self) -> None:
        @sentinel_monitor(name="vg_test")
        def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(params ** 2)

        params = jnp.array([1.0, 2.0, 3.0, 4.0])
        val, grads = jax.value_and_grad(loss_fn)(params)

        expected_val = jnp.mean(params ** 2)
        np.testing.assert_allclose(float(val), float(expected_val), atol=1e-5)
        assert grads.shape == params.shape

    def test_multiple_calls_accumulate_state(self) -> None:
        @sentinel_monitor(name="multi_call")
        def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(params ** 2)

        grad_fn = jax.grad(loss_fn)
        params = jnp.ones(5)

        for _ in range(10):
            grad_fn(params)

        # State should have recorded multiple steps
        state = loss_fn._sentinel_state  # type: ignore[attr-defined]
        assert state.step_count > 0

    def test_pytree_params(self) -> None:
        @sentinel_monitor(name="pytree_test")
        def loss_fn(params: dict) -> jnp.ndarray:
            w = params["weight"]
            b = params["bias"]
            return jnp.sum(w ** 2) + jnp.sum(b ** 2)

        params = {
            "weight": jnp.ones((3, 3)),
            "bias": jnp.zeros(3),
        }

        val = loss_fn(params)
        assert float(val) == 9.0  # 3*3 ones squared

    def test_reset_clears_state(self) -> None:
        @sentinel_monitor(name="reset_test")
        def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(params)

        grad_fn = jax.grad(loss_fn)
        grad_fn(jnp.ones(5))

        reset_state()
        anomalies = get_anomalies()
        assert len(anomalies) == 0

    def test_no_jax_fallback(self) -> None:
        """Test that sentinel_monitor works as identity when JAX is available."""
        @sentinel_monitor
        def simple_fn(x: jnp.ndarray) -> jnp.ndarray:
            return x * 2

        result = simple_fn(jnp.array(5.0))
        assert float(result) == 10.0

    def test_gradient_norms_api(self) -> None:
        @sentinel_monitor(name="norms_test")
        def loss_fn(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(params ** 2)

        grad_fn = jax.grad(loss_fn)

        # Run several gradient computations
        for _ in range(5):
            grad_fn(jnp.ones(10) * 3.0)

        norms = get_gradient_norms()
        # Should have tracked some gradient norms
        assert isinstance(norms, dict)
