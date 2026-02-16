"""Tests for pyneuroglm.regression.likelihood."""

import numpy as np
import pytest

from pyneuroglm.regression.likelihood import poisson as poisson_loglik
from pyneuroglm.regression.nonlinearity import exp as exp_nonlinearity

TOL_LOGLIK = 1e-8
TOL_GRADIENT = 1e-8
TOL_HESSIAN = 1e-7


class TestPoissonLikelihoodParity:
    """Poisson likelihood parity tests against MATLAB reference implementation."""

    @pytest.fixture
    def data(self, matlab_regression_data):
        return matlab_regression_data

    def test_likelihood_at_zero_weights(self, data, reference_poisson_negloglik):
        """Verify likelihood, gradient, and Hessian parity at w = 0."""
        X, y = data
        w = np.zeros(X.shape[1])

        L_ref, dL_ref, H_ref = reference_poisson_negloglik(w, X, y)
        L_py, dL_py, H_py = poisson_loglik(w, X, y, exp_nonlinearity)

        assert np.isclose(L_py, -L_ref, atol=TOL_LOGLIK)
        assert np.max(np.abs(dL_py - (-dL_ref))) < TOL_GRADIENT
        assert np.max(np.abs(H_py - (-H_ref))) < TOL_HESSIAN

    def test_likelihood_at_random_weights(self, data, reference_poisson_negloglik):
        """Verify parity at random (small) weights."""
        X, y = data
        rng = np.random.default_rng(42)
        w = rng.standard_normal(X.shape[1]) * 0.01

        L_ref, dL_ref, H_ref = reference_poisson_negloglik(w, X, y)
        L_py, dL_py, H_py = poisson_loglik(w, X, y, exp_nonlinearity)

        assert np.isclose(L_py, -L_ref, atol=TOL_LOGLIK)
        assert np.max(np.abs(dL_py - (-dL_ref))) < TOL_GRADIENT
        assert np.max(np.abs(H_py - (-H_ref))) < TOL_HESSIAN

    def test_gradient_finite_difference(self, data):
        """Verify gradient via central finite differences."""
        X, y = data
        w = np.zeros(X.shape[1])

        _, dL, _ = poisson_loglik(w, X, y, exp_nonlinearity)

        eps = 1e-6
        dL_fd = np.zeros_like(w)
        for i in range(len(w)):
            w_plus = w.copy(); w_plus[i] += eps
            w_minus = w.copy(); w_minus[i] -= eps
            L_plus = poisson_loglik(w_plus, X, y, exp_nonlinearity)[0]
            L_minus = poisson_loglik(w_minus, X, y, exp_nonlinearity)[0]
            dL_fd[i] = (L_plus - L_minus) / (2 * eps)

        rel_diff = np.max(np.abs(dL - dL_fd)) / (np.max(np.abs(dL)) + 1e-10)
        assert rel_diff < 1e-4

    def test_hessian_finite_difference(self, data):
        """Verify Hessian via finite differences on gradient (subset of entries)."""
        X, y = data
        w = np.zeros(X.shape[1])

        _, _, H = poisson_loglik(w, X, y, exp_nonlinearity)

        eps = 1e-6
        check = list(range(min(10, len(w))))
        max_diff = 0.0
        for i in check:
            w_plus = w.copy(); w_plus[i] += eps
            w_minus = w.copy(); w_minus[i] -= eps
            dL_plus = poisson_loglik(w_plus, X, y, exp_nonlinearity)[1]
            dL_minus = poisson_loglik(w_minus, X, y, exp_nonlinearity)[1]
            H_fd_row = (dL_plus - dL_minus) / (2 * eps)
            for j in check:
                max_diff = max(max_diff, abs(H[i, j] - H_fd_row[j]))

        assert max_diff < 1e-3
