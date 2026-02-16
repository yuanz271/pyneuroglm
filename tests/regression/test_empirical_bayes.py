"""Tests for pyneuroglm.regression.empirical_bayes."""

import numpy as np
import pytest
from scipy.special import logsumexp

from pyneuroglm.regression.empirical_bayes import log_evidence
from pyneuroglm.regression.likelihood import poisson as poisson_loglik
from pyneuroglm.regression.nonlinearity import exp as exp_nlfun
from pyneuroglm.regression.posterior import get_posterior_weights, initialize_lstsq
from pyneuroglm.regression.prior import gaussian_zero_mean_inv


def _gaussian_loglik_factory(X, y, sigma2):
    """Create log-likelihood function for Gaussian model."""
    XtX = X.T @ X
    Xty = X.T @ y

    def loglik(w, *args):
        r = y - X @ w
        L = -0.5 / sigma2 * (r @ r)
        dL = (Xty - XtX @ w) / sigma2
        ddL = -XtX / sigma2
        return L, dL, ddL

    return loglik


@pytest.mark.parametrize(
    "seed, n, m, sigma2, alpha",
    [
        (0, 50, 4, 0.5, 2.0),
        (42, 200, 20, 1.0, 1.0),
        (123, 100, 8, 0.25, 50.0),
    ],
    ids=["small", "larger", "strong-prior"],
)
def test_laplace_exact_for_gaussian(seed, n, m, sigma2, alpha):
    """Laplace approximation should be exact for Gaussian models."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m))
    w_true = rng.standard_normal(m)
    y = X @ w_true + rng.standard_normal(n) * np.sqrt(sigma2)

    Cinv = alpha * np.eye(m)
    loglik = _gaussian_loglik_factory(X, y, sigma2)

    A = (X.T @ X) / sigma2 + Cinv
    b = (X.T @ y) / sigma2
    w_map = np.linalg.solve(A, b)

    L_map = loglik(w_map)[0]
    P_map = gaussian_zero_mean_inv(w_map, Cinv)[0]
    _, logdetA = np.linalg.slogdet(A)
    logZ_exact = L_map + P_map + 0.5 * m * np.log(2 * np.pi) - 0.5 * logdetA

    logZ_code = log_evidence(
        param=w_map, hyperparam=Cinv,
        loglik=loglik, llargs=(),
        logprior=gaussian_zero_mean_inv, lpargs=(),
    )
    # log_evidence omits the (m/2)log(2pi) term
    logZ_code_full = logZ_code + 0.5 * m * np.log(2 * np.pi)

    assert np.isclose(logZ_code_full, logZ_exact, atol=1e-10)


@pytest.mark.parametrize(
    "seed, n, w_true, alpha, tol",
    [
        (0, 200, np.array([0.3]), 2.0, 0.05),
        (1, 300, np.array([0.5]), 0.5, 0.1),
        (2, 200, np.array([0.2]), 10.0, 0.05),
    ],
    ids=["moderate", "weak-prior", "strong-prior"],
)
def test_laplace_close_to_quadrature_poisson_1d(seed, n, w_true, alpha, tol):
    """Laplace approximation should be close to 1D numerical quadrature for Poisson."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 1))
    rate = np.exp(X @ w_true).ravel()
    y = rng.poisson(rate)

    Cinv = alpha * np.eye(1)

    w_map, _, _ = get_posterior_weights(
        X, y, Cinv, dist="poisson", initialize=initialize_lstsq, init_kwargs={},
    )

    _, _, ddL = poisson_loglik(w_map, X, y, exp_nlfun)
    _, _, ddP = gaussian_zero_mean_inv(w_map, Cinv)
    H = -(ddL + ddP)
    std = 1.0 / np.sqrt(H[0, 0])

    w0 = float(w_map[0])
    grid = np.linspace(w0 - 12 * std, w0 + 12 * std, 20001)
    dw = grid[1] - grid[0]

    logj = np.array([
        poisson_loglik(np.array([wi]), X, y, exp_nlfun)[0]
        + gaussian_zero_mean_inv(np.array([wi]), Cinv)[0]
        for wi in grid
    ])
    logZ_numeric = logsumexp(logj) + np.log(dw)

    logZ_laplace = log_evidence(
        param=w_map, hyperparam=Cinv,
        loglik=lambda w, *args: poisson_loglik(w, X, y, exp_nlfun),
        llargs=(), logprior=gaussian_zero_mean_inv, lpargs=(),
    )
    logZ_laplace_full = float(logZ_laplace + 0.5 * np.log(2 * np.pi))

    assert abs(logZ_laplace_full - logZ_numeric) < tol
