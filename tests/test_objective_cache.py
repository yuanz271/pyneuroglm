"""Tests for the Objective cache in pyneuroglm.regression.optim."""

import numpy as np
from scipy.optimize import minimize

from pyneuroglm.regression.optim import Objective


def _quadratic(x):
    """Quadratic objective returning (value, gradient, Hessian)."""
    v = float(np.sum(x**2))
    g = 2 * x.copy()
    H = 2 * np.eye(len(x))
    return v, g, H


def test_cache_invalidates_on_inplace_mutation():
    """Cache must recompute when the same array object is mutated in place."""
    obj = Objective(_quadratic)

    x = np.array([1.0, 2.0])
    v1 = obj.function(x)
    assert v1 == 5.0

    # Mutate in place — simulates scipy reusing the same array object
    x[:] = [3.0, 4.0]
    v2 = obj.function(x)
    assert v2 == 25.0


def test_cache_hit_same_values():
    """Cache should return cached result when called with equal values."""
    call_count = 0

    def counting_quad(x):
        nonlocal call_count
        call_count += 1
        return _quadratic(x)

    obj = Objective(counting_quad)

    x = np.array([1.0, 2.0])
    obj.function(x)
    assert call_count == 1

    # Same values, different object — should be a cache hit
    obj.gradient(np.array([1.0, 2.0]))
    assert call_count == 1

    obj.hessian(np.array([1.0, 2.0]))
    assert call_count == 1


def test_cache_miss_different_values():
    """Cache should recompute when called with different values."""
    call_count = 0

    def counting_quad(x):
        nonlocal call_count
        call_count += 1
        return _quadratic(x)

    obj = Objective(counting_quad)

    obj.function(np.array([1.0, 2.0]))
    assert call_count == 1

    obj.function(np.array([3.0, 4.0]))
    assert call_count == 2


def test_flip_sign():
    """flip_sign=True should negate all returned values."""
    obj = Objective(_quadratic, flip_sign=True)

    x = np.array([1.0, 2.0])
    assert obj.function(x) == -5.0
    np.testing.assert_array_equal(obj.gradient(x), np.array([-2.0, -4.0]))
    np.testing.assert_array_equal(obj.hessian(x), -2 * np.eye(2))


def test_optimizer_convergence_with_cache():
    """Objective cache should not break scipy trust-ncg convergence."""

    def rosenbrock(x):
        v = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        g = np.array(
            [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]
        )
        H = np.array(
            [[-400 * (x[1] - 3 * x[0] ** 2) + 2, -400 * x[0]], [-400 * x[0], 200]]
        )
        return float(v), g, H

    obj = Objective(rosenbrock, flip_sign=False)
    result = minimize(
        obj.function,
        np.array([0.5, 0.5]),
        method="trust-ncg",
        jac=obj.gradient,
        hess=obj.hessian,
        options={"gtol": 1e-8},
    )

    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-5)
