"""
Ground-truth validation tests for Laplace log-evidence approximation.

These tests compare log_evidence() against independent reference values:
1. Gaussian model: Laplace is exact (closed-form evidence)
2. Poisson 1D: numerical quadrature (independent integral)
"""

import numpy as np
from scipy.special import logsumexp

from pyneuroglm.regression.empirical_bayes import log_evidence
from pyneuroglm.regression.likelihood import poisson as poisson_loglik
from pyneuroglm.regression.nonlinearity import exp as exp_nlfun
from pyneuroglm.regression.posterior import get_posterior_weights, initialize_lstsq
from pyneuroglm.regression.prior import gaussian_zero_mean_inv


def gaussian_loglik_factory(X, y, sigma2):
    """
    Create log-likelihood function for Gaussian model (Bayesian linear regression).

    L(w) = -0.5/sigma2 * ||y - Xw||^2  (unnormalized, no constants)

    This matches the convention in pyneuroglm where likelihoods omit constants.
    """
    XtX = X.T @ X
    Xty = X.T @ y

    def loglik(w, *args):
        r = y - X @ w
        L = -0.5 / sigma2 * (r @ r)
        dL = (Xty - XtX @ w) / sigma2
        ddL = -XtX / sigma2
        return L, dL, ddL

    return loglik


class TestLaplaceExactGaussian:
    """
    Test Laplace approximation on Gaussian model where it should be exact.

    For quadratic log-joint (Gaussian likelihood + Gaussian prior), the Laplace
    approximation equals the true integral exactly. This validates the formula
    and sign conventions in log_evidence().
    """

    def test_laplace_exact_for_gaussian_small(self):
        """Small problem (m=4) with moderate condition number."""
        rng = np.random.default_rng(0)
        n, m = 50, 4
        X = rng.standard_normal((n, m))
        w_true = rng.standard_normal(m)
        sigma2 = 0.5
        y = X @ w_true + rng.standard_normal(n) * np.sqrt(sigma2)

        alpha = 2.0
        Cinv = alpha * np.eye(m)

        self._run_gaussian_exactness_check(X, y, sigma2, Cinv, m)

    def test_laplace_exact_for_gaussian_larger(self):
        """Larger problem (m=20) to stress-test numerical stability."""
        rng = np.random.default_rng(42)
        n, m = 200, 20
        X = rng.standard_normal((n, m))
        w_true = rng.standard_normal(m)
        sigma2 = 1.0
        y = X @ w_true + rng.standard_normal(n) * np.sqrt(sigma2)

        alpha = 1.0
        Cinv = alpha * np.eye(m)

        self._run_gaussian_exactness_check(X, y, sigma2, Cinv, m)

    def test_laplace_exact_for_gaussian_strong_prior(self):
        """Strong prior (large alpha) shrinks posterior toward zero."""
        rng = np.random.default_rng(123)
        n, m = 100, 8
        X = rng.standard_normal((n, m))
        w_true = rng.standard_normal(m) * 0.5
        sigma2 = 0.25
        y = X @ w_true + rng.standard_normal(n) * np.sqrt(sigma2)

        alpha = 50.0
        Cinv = alpha * np.eye(m)

        self._run_gaussian_exactness_check(X, y, sigma2, Cinv, m)

    def _run_gaussian_exactness_check(self, X, y, sigma2, Cinv, m):
        loglik = gaussian_loglik_factory(X, y, sigma2)

        # Closed-form MAP: w_MAP = (X'X/σ² + Cinv)^{-1} X'y/σ²
        A = (X.T @ X) / sigma2 + Cinv
        b = (X.T @ y) / sigma2
        w_map = np.linalg.solve(A, b)

        # Exact log-evidence for Gaussian (unnormalized L and P):
        # log Z = ℓ(w_MAP) + (m/2)log(2π) - (1/2)log|A|
        L_map = loglik(w_map)[0]
        P_map = gaussian_zero_mean_inv(w_map, Cinv)[0]
        ell_map = L_map + P_map

        sign, logdetA = np.linalg.slogdet(A)
        assert sign > 0, "A should be positive definite"

        logZ_exact = ell_map + 0.5 * m * np.log(2 * np.pi) - 0.5 * logdetA

        # log_evidence() omits the (m/2)log(2π) term
        logZ_code = log_evidence(
            param=w_map,
            hyperparam=Cinv,
            loglik=loglik,
            llargs=(),
            logprior=gaussian_zero_mean_inv,
            lpargs=(),
        )

        logZ_code_with_constant = logZ_code + 0.5 * m * np.log(2 * np.pi)

        assert np.isclose(logZ_code_with_constant, logZ_exact, atol=1e-10), (
            f"Laplace should be exact for Gaussian: "
            f"code={logZ_code_with_constant:.10f}, exact={logZ_exact:.10f}, "
            f"diff={abs(logZ_code_with_constant - logZ_exact):.2e}"
        )


class TestLaplaceQuadraturePoisson1D:
    """
    Test Laplace approximation against 1D numerical quadrature for Poisson GLM.

    For a single-parameter Poisson GLM, we can compute the evidence integral
    numerically on a grid. This provides an independent ground-truth that
    does not share the same code path as log_evidence().
    """

    def test_laplace_close_to_quadrature_poisson_1d(self):
        """Standard 1D Poisson GLM with moderate regularization."""
        rng = np.random.default_rng(0)
        n = 200
        X = rng.standard_normal((n, 1))
        w_true = np.array([0.3])
        rate = np.exp(X @ w_true).ravel()
        y = rng.poisson(rate)

        alpha = 2.0
        Cinv = alpha * np.eye(1)

        diff = self._run_poisson_1d_quadrature_check(X, y, Cinv)
        assert diff < 0.05, f"Laplace vs quadrature mismatch: {diff:.4f}"

    def test_laplace_close_to_quadrature_poisson_1d_weak_prior(self):
        """Weak prior (small alpha) gives broader posterior."""
        rng = np.random.default_rng(1)
        n = 300
        X = rng.standard_normal((n, 1))
        w_true = np.array([0.5])
        rate = np.exp(X @ w_true).ravel()
        y = rng.poisson(rate)

        alpha = 0.5
        Cinv = alpha * np.eye(1)

        diff = self._run_poisson_1d_quadrature_check(X, y, Cinv)
        assert diff < 0.1, f"Laplace vs quadrature mismatch: {diff:.4f}"

    def test_laplace_close_to_quadrature_poisson_1d_strong_prior(self):
        """Strong prior (large alpha) gives tighter posterior."""
        rng = np.random.default_rng(2)
        n = 200
        X = rng.standard_normal((n, 1))
        w_true = np.array([0.2])
        rate = np.exp(X @ w_true).ravel()
        y = rng.poisson(rate)

        alpha = 10.0
        Cinv = alpha * np.eye(1)

        diff = self._run_poisson_1d_quadrature_check(X, y, Cinv)
        assert diff < 0.05, f"Laplace vs quadrature mismatch: {diff:.4f}"

    def _run_poisson_1d_quadrature_check(self, X, y, Cinv):
        w_map, _, _ = get_posterior_weights(
            X,
            y,
            Cinv,
            dist="poisson",
            cvfolds=None,
            initialize=initialize_lstsq,
            init_kwargs={},
        )

        _, _, ddL = poisson_loglik(w_map, X, y, exp_nlfun)
        _, _, ddP = gaussian_zero_mean_inv(w_map, Cinv)

        H = -(ddL + ddP)
        assert H[0, 0] > 0, "Negative Hessian should be positive"
        std = 1.0 / np.sqrt(H[0, 0])

        w0 = float(w_map[0])
        lo, hi = w0 - 12 * std, w0 + 12 * std
        grid = np.linspace(lo, hi, 20001)
        dw = grid[1] - grid[0]

        logj = np.array(
            [
                poisson_loglik(np.array([wi]), X, y, exp_nlfun)[0]
                + gaussian_zero_mean_inv(np.array([wi]), Cinv)[0]
                for wi in grid
            ]
        )

        logZ_numeric = logsumexp(logj) + np.log(dw)

        logZ_laplace = log_evidence(
            param=w_map,
            hyperparam=Cinv,
            loglik=lambda w, *args: poisson_loglik(w, X, y, exp_nlfun),
            llargs=(),
            logprior=gaussian_zero_mean_inv,
            lpargs=(),
        )

        m = 1
        logZ_laplace_full = float(logZ_laplace + 0.5 * m * np.log(2 * np.pi))

        return abs(logZ_laplace_full - logZ_numeric)
