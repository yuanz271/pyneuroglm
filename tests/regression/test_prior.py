"""Tests for pyneuroglm.regression.prior."""

import numpy as np

from pyneuroglm.regression.prior import ridge_Cinv, gaussian_zero_mean_inv


def test_ridge_cinv():
    """Verify ridge_Cinv produces rho * I and zeros out intercept when requested."""
    rho = 2.5
    nx = 10

    # Without intercept
    Cinv = ridge_Cinv(rho, nx)
    assert np.allclose(Cinv, rho * np.eye(nx))

    # With intercept
    Cinv = ridge_Cinv(rho, nx, intercept_prepended=True)
    assert Cinv[0, 0] == 0
    assert np.allclose(np.diag(Cinv)[1:], rho)


def test_gaussian_prior_parity():
    """Verify gaussian_zero_mean_inv matches MATLAB sign convention."""
    rng = np.random.default_rng(789)
    nx = 15
    rho = 1.5
    w = rng.standard_normal(nx)
    Cinv = ridge_Cinv(rho, nx)

    P, dP, ddP = gaussian_zero_mean_inv(w, Cinv)

    # MATLAB returns negative log-prior; Python returns positive
    p_matlab = 0.5 * w @ Cinv @ w
    assert np.isclose(P, -p_matlab, atol=1e-10)
    assert np.allclose(dP, -Cinv @ w, atol=1e-10)
    assert np.allclose(ddP, -Cinv, atol=1e-10)
