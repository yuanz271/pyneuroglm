"""
Poisson likelihood parity test between MATLAB neuroGLM and pyneuroglm.

This test validates that pyneuroglm's Poisson log-likelihood, gradient, and Hessian
match MATLAB's glms.neglog.poisson (with appropriate sign conventions).

Reference:
- MATLAB source: neuroGLM/matRegress/+glms/+neglog/poisson.m
- MATLAB returns NEGATIVE log-likelihood; Python returns POSITIVE log-likelihood
- Parity: L_python == -L_matlab, dL_python == -dL_matlab, H_python == -H_matlab

Sign Convention:
- MATLAB: minimizes negative log-likelihood (L > 0 for valid data)
- Python: maximizes log-likelihood (L < 0 for valid data, typically)
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from pyneuroglm import Experiment, Trial
from pyneuroglm.design import DesignMatrix
from pyneuroglm.regression.likelihood import poisson as poisson_loglik
from pyneuroglm.regression.nonlinearity import exp as exp_nonlinearity


TOL_LOGLIK = 1e-8
TOL_GRADIENT = 1e-8
TOL_HESSIAN = 1e-7

MATLAB_DIR = Path(__file__).parent.parent / "neuroGLM"
EXAMPLE_DIR = Path(__file__).parent.parent / "example"


def reference_poisson_negloglik(w, X, y):
    """
    Compute MATLAB-equivalent Poisson negative log-likelihood.

    Computes NEGATIVE log-likelihood, gradient, and Hessian for Poisson GLM
    with canonical exp link (no dt scaling, matching MATLAB exactly).

    This is a pure NumPy implementation of:
        neuroGLM/matRegress/+glms/+neglog/poisson.m

    Parameters
    ----------
    w : ndarray of shape (p,)
        Weight vector.
    X : ndarray of shape (n, p)
        Design matrix.
    y : ndarray of shape (n,)
        Spike counts (non-negative integers).

    Returns
    -------
    L : float
        Negative log-likelihood.
    dL : ndarray of shape (p,)
        Gradient of negative log-likelihood.
    H : ndarray of shape (p, p)
        Hessian of negative log-likelihood.

    Notes
    -----
    MATLAB formula (line 37, 57):
        L = -y'*log(f) + sum(f)   where f = exp(X*w)

    Gradient (line 61):
        dL = X' * ((1 - y./f) .* df)   where df = f for exp link

    Hessian (line 63-64):
        h = ddf.*(1-y./f) + y.*(df./f).^2   where ddf = f for exp link
        H = X' * diag(h) * X

    For exp link: f = df = ddf = exp(eta), so:
        h = f*(1 - y/f) + y*(f/f)^2 = f - y + y = f
        H = X' * diag(f) * X
    """
    eta = X @ w
    f = np.exp(eta)
    nz = f > 0

    if np.any(y[~nz] != 0):
        return np.inf, np.full_like(w, np.nan), np.full((len(w), len(w)), np.nan)

    # MATLAB line 37/57: L = -y'*log(f) + sum(f)
    L = -np.sum(y[nz] * np.log(f[nz])) + np.sum(f)

    # MATLAB line 61: dL = X' * ((1 - y./f) .* df) → X' * (f - y) for exp link
    d = f[nz] - y[nz]
    dL = X[nz].T @ d

    # MATLAB lines 63-64: h = f for exp link, H = X' * diag(h) * X
    h = f[nz]
    H = np.einsum("ij,i,ik->jk", X[nz], h, X[nz])

    return L, dL, H


def load_design_matrix_and_spikes():
    """
    Load the validated design matrix and compute spike counts.

    Uses the same setup as test_design_matrix_parity.py to ensure
    we're testing on data where design matrix parity is already proven.

    Returns
    -------
    X : ndarray of shape (n_bins, n_features)
        Design matrix (z-scored LFP columns, no bias).
    y : ndarray of shape (n_bins,)
        Binned spike counts.
    """
    # Load MATLAB design matrix (already z-scored, no bias)
    dm_path = MATLAB_DIR / "exampleDM.mat"
    if not dm_path.exists():
        # Fallback to example dir
        dm_path = EXAMPLE_DIR / "exampleDM.mat"

    mat = sio.loadmat(str(dm_path), squeeze_me=True, struct_as_record=False)
    X = mat["Z"]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    # Load trial data and compute binned spikes
    data_path = MATLAB_DIR / "exampleData.mat"
    if not data_path.exists():
        data_path = EXAMPLE_DIR / "exampleData.mat"

    mat = sio.loadmat(str(data_path), squeeze_me=True, struct_as_record=False)
    trials_raw = mat["trial"]

    # Build experiment to use get_binned_spike
    expt = Experiment(time_unit="ms", binsize=1, eid="parity")
    expt.register_spike_train("sptrain", "Our Neuron")

    trial_indices = list(range(10))
    for i in trial_indices:
        t = trials_raw[i]
        trial = Trial(tid=i, duration=int(t.duration))
        trial["sptrain"] = np.asarray(t.sptrain, dtype=np.float64)
        expt.trials[i] = trial

    # Create minimal design matrix just to get binned spikes
    dm = DesignMatrix(expt)
    y = dm.get_binned_spike("sptrain", trial_indices)

    assert X.shape[0] == len(y), f"Shape mismatch: X has {X.shape[0]} rows, y has {len(y)} elements"

    return X, y


class TestPoissonLikelihoodParity:
    """Poisson likelihood parity test suite."""

    @pytest.fixture
    def data(self):
        """Load design matrix and spike counts."""
        dm_path = MATLAB_DIR / "exampleDM.mat"
        data_path = MATLAB_DIR / "exampleData.mat"
        if not dm_path.exists() or not data_path.exists():
            pytest.skip("MATLAB fixtures not found")
        return load_design_matrix_and_spikes()

    def test_likelihood_at_zero_weights(self, data):
        """
        Test 1: Verify likelihood parity at w = 0.

        At w=0, eta=0, lambda=1 for all bins.
        This is a simple, well-defined test point.
        """
        X, y = data
        p = X.shape[1]
        w = np.zeros(p)

        # Reference (MATLAB-style negative log-likelihood)
        L_ref, dL_ref, H_ref = reference_poisson_negloglik(w, X, y)

        # Python (positive log-likelihood)
        L_py, dL_py, H_py = poisson_loglik(w, X, y, exp_nonlinearity)

        print("\n=== Likelihood at w=0 ===")
        print(f"Reference L (neg): {L_ref:.6f}")
        print(f"Python L (pos):    {L_py:.6f}")
        print(f"Expected -L_ref:   {-L_ref:.6f}")
        print(f"L difference:      {abs(L_py - (-L_ref)):.2e}")

        # L_python should equal -L_matlab
        assert np.isclose(L_py, -L_ref, atol=TOL_LOGLIK), (
            f"Log-likelihood mismatch: Python={L_py:.8f}, -MATLAB={-L_ref:.8f}, "
            f"diff={abs(L_py - (-L_ref)):.2e}"
        )

        # Gradient check
        grad_diff = np.max(np.abs(dL_py - (-dL_ref)))
        print(f"Gradient max diff: {grad_diff:.2e}")
        assert grad_diff < TOL_GRADIENT, f"Gradient mismatch: max diff = {grad_diff:.2e}"

        # Hessian check
        hess_diff = np.max(np.abs(H_py - (-H_ref)))
        print(f"Hessian max diff:  {hess_diff:.2e}")
        assert hess_diff < TOL_HESSIAN, f"Hessian mismatch: max diff = {hess_diff:.2e}"

        print("PASSED: Likelihood parity at w=0")

    def test_likelihood_at_random_weights(self, data):
        """
        Test 2: Verify likelihood parity at random weights.

        Uses seeded RNG for reproducibility.
        Small weights to keep rates reasonable.
        """
        X, y = data
        p = X.shape[1]

        # Seeded random weights (small magnitude to avoid overflow)
        rng = np.random.default_rng(42)
        w = rng.standard_normal(p) * 0.01

        # Reference
        L_ref, dL_ref, H_ref = reference_poisson_negloglik(w, X, y)

        # Python
        L_py, dL_py, H_py = poisson_loglik(w, X, y, exp_nonlinearity)

        print("\n=== Likelihood at random w (seed=42) ===")
        print(f"Reference L (neg): {L_ref:.6f}")
        print(f"Python L (pos):    {L_py:.6f}")
        print(f"L difference:      {abs(L_py - (-L_ref)):.2e}")

        # Log-likelihood
        assert np.isclose(
            L_py, -L_ref, atol=TOL_LOGLIK
        ), f"Log-likelihood mismatch: Python={L_py:.8f}, -MATLAB={-L_ref:.8f}"

        # Gradient
        grad_diff = np.max(np.abs(dL_py - (-dL_ref)))
        print(f"Gradient max diff: {grad_diff:.2e}")
        assert grad_diff < TOL_GRADIENT, f"Gradient mismatch: max diff = {grad_diff:.2e}"

        # Hessian
        hess_diff = np.max(np.abs(H_py - (-H_ref)))
        print(f"Hessian max diff:  {hess_diff:.2e}")
        assert hess_diff < TOL_HESSIAN, f"Hessian mismatch: max diff = {hess_diff:.2e}"

        print("PASSED: Likelihood parity at random w")

    def test_gradient_finite_difference(self, data):
        """
        Test 3: Verify gradient via finite differences.

        This is a sanity check independent of MATLAB reference.
        """
        X, y = data
        p = X.shape[1]

        w = np.zeros(p)

        L, dL, _ = poisson_loglik(w, X, y, exp_nonlinearity)

        eps = 1e-6
        dL_fd = np.zeros(p)
        for i in range(p):
            w_plus = w.copy()
            w_plus[i] += eps
            w_minus = w.copy()
            w_minus[i] -= eps
            L_plus, _, _ = poisson_loglik(w_plus, X, y, exp_nonlinearity)
            L_minus, _, _ = poisson_loglik(w_minus, X, y, exp_nonlinearity)
            dL_fd[i] = (L_plus - L_minus) / (2 * eps)

        grad_diff = np.max(np.abs(dL - dL_fd))
        rel_diff = grad_diff / (np.max(np.abs(dL)) + 1e-10)
        print("\n=== Gradient finite difference check ===")
        print(f"Max abs diff: {grad_diff:.2e}, relative: {rel_diff:.2e}")

        assert rel_diff < 1e-4, f"Gradient FD mismatch: relative diff = {rel_diff:.2e}"

        print("PASSED: Gradient finite difference check")

    def test_hessian_finite_difference(self, data):
        """
        Test 4: Verify Hessian via finite differences on gradient.

        This is a sanity check independent of MATLAB reference.
        Checks a subset of Hessian entries for efficiency.
        """
        X, y = data
        p = X.shape[1]

        w = np.zeros(p)

        _, _, H = poisson_loglik(w, X, y, exp_nonlinearity)

        eps = 1e-6
        check_indices = list(range(min(10, p)))

        max_diff = 0.0
        for i in check_indices:
            w_plus = w.copy()
            w_plus[i] += eps
            w_minus = w.copy()
            w_minus[i] -= eps
            _, dL_plus, _ = poisson_loglik(w_plus, X, y, exp_nonlinearity)
            _, dL_minus, _ = poisson_loglik(w_minus, X, y, exp_nonlinearity)
            H_fd_row = (dL_plus - dL_minus) / (2 * eps)

            for j in check_indices:
                diff = abs(H[i, j] - H_fd_row[j])
                max_diff = max(max_diff, diff)

        print("\n=== Hessian finite difference check ===")
        print(f"Max diff (analytical vs FD): {max_diff:.2e}")

        # Finite difference tolerance for Hessian is looser
        assert max_diff < 1e-3, f"Hessian finite difference mismatch: max diff = {max_diff:.2e}"

        print("PASSED: Hessian finite difference check")


if __name__ == "__main__":
    print("Loading data...")
    X, y = load_design_matrix_and_spikes()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}, sum: {y.sum()}, range: [{y.min()}, {y.max()}]")

    print("\n" + "=" * 60)
    print("Testing at w = 0")
    print("=" * 60)

    p = X.shape[1]
    w = np.zeros(p)

    L_ref, dL_ref, H_ref = reference_poisson_negloglik(w, X, y)
    L_py, dL_py, H_py = poisson_loglik(w, X, y, exp_nonlinearity)

    print(f"Reference L (neg): {L_ref:.6f}")
    print(f"Python L (pos):    {L_py:.6f}")
    print(f"L diff from parity: {abs(L_py - (-L_ref)):.2e}")
    print(f"Gradient max diff:  {np.max(np.abs(dL_py - (-dL_ref))):.2e}")
    print(f"Hessian max diff:   {np.max(np.abs(H_py - (-H_ref))):.2e}")


class TestPriorParity:
    """Prior parity test suite (ridge/Gaussian zero-mean)."""

    def test_ridge_cinv_matches_matlab(self):
        """Verify ridge_Cinv produces rho * I (MATLAB: speye(nx)*rho)."""
        from pyneuroglm.regression.prior import ridge_Cinv

        rho = 2.5
        nx = 10

        Cinv = ridge_Cinv(rho, nx)

        expected = rho * np.eye(nx)
        assert np.allclose(Cinv, expected), "ridge_Cinv should produce rho * I"

    def test_ridge_cinv_intercept_handling(self):
        """Verify intercept is not regularized when intercept_prepended=True."""
        from pyneuroglm.regression.prior import ridge_Cinv

        rho = 2.5
        nx = 10

        Cinv = ridge_Cinv(rho, nx, intercept_prepended=True)

        assert Cinv[0, 0] == 0, "Intercept should not be regularized"
        assert np.allclose(np.diag(Cinv)[1:], rho), "Other weights should have rho"

    def test_gaussian_prior_parity(self):
        """
        Verify gaussian_zero_mean_inv matches MATLAB gpriors.gaussian_zero_mean_inv.

        MATLAB returns NEGATIVE log-prior; Python returns POSITIVE log-prior.
        """
        from pyneuroglm.regression.prior import gaussian_zero_mean_inv, ridge_Cinv

        rng = np.random.default_rng(789)
        nx = 15
        rho = 1.5
        w = rng.standard_normal(nx)

        Cinv = ridge_Cinv(rho, nx)

        P_py, dP_py, ddP_py = gaussian_zero_mean_inv(w, Cinv)

        p_matlab = 0.5 * w @ Cinv @ w
        dp_matlab = Cinv @ w
        ddp_matlab = Cinv

        assert np.isclose(
            P_py, -p_matlab, atol=1e-10
        ), f"Prior value mismatch: Python={P_py}, -MATLAB={-p_matlab}"
        assert np.allclose(dP_py, -dp_matlab, atol=1e-10), "Prior gradient mismatch"
        assert np.allclose(ddP_py, -ddp_matlab, atol=1e-10), "Prior Hessian mismatch"

        print("\n=== Prior Parity Test ===")
        print(f"MATLAB p (neg log-prior): {p_matlab:.6f}")
        print(f"Python P (log-prior):     {P_py:.6f}")
        print(f"Parity check: P_py == -p_matlab: {np.isclose(P_py, -p_matlab)}")
        print("PASSED: Prior parity verified")


class TestMAPRegression:
    """MAP regression end-to-end test using BayesianGLMRegressor."""

    @pytest.fixture
    def regression_data(self):
        """Load design matrix and spike counts for regression."""
        dm_path = MATLAB_DIR / "exampleDM.mat"
        data_path = MATLAB_DIR / "exampleData.mat"
        if not dm_path.exists() or not data_path.exists():
            pytest.skip("MATLAB fixtures not found")
        return load_design_matrix_and_spikes()

    def test_bayesian_glm_convergence(self, regression_data):
        """Verify BayesianGLMRegressor converges on validated data."""
        from pyneuroglm.regression.sklearn import BayesianGLMRegressor

        X, y = regression_data

        model = BayesianGLMRegressor(alpha=1.0, fit_intercept=True, initialize="lstsq")
        model.fit(X, y)

        assert hasattr(model, "coef_"), "Model should have coef_ after fit"
        assert hasattr(model, "intercept_"), "Model should have intercept_ after fit"
        assert len(model.coef_) == X.shape[1], "coef_ length should match features"

        print("\n=== MAP Regression Convergence ===")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"Intercept: {model.intercept_:.4f}")
        print(f"Coef range: [{model.coef_.min():.4f}, {model.coef_.max():.4f}]")
        print("PASSED: Model converged")

    def test_first_order_optimality(self, regression_data):
        """Verify gradient is near zero at MAP solution (first-order condition)."""
        from pyneuroglm.regression.sklearn import BayesianGLMRegressor
        from pyneuroglm.regression.posterior import poisson as poisson_posterior
        from pyneuroglm.regression.prior import ridge_Cinv
        from pyneuroglm.regression.nonlinearity import exp as exp_nlfun

        X, y = regression_data

        model = BayesianGLMRegressor(alpha=1.0, fit_intercept=True, initialize="lstsq")
        model.fit(X, y)

        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        w = np.concatenate([[model.intercept_], model.coef_])
        Cinv = ridge_Cinv(model.alpha, X_with_intercept.shape[1], intercept_prepended=True)

        _, grad, _ = poisson_posterior(w, X_with_intercept, y, Cinv, exp_nlfun, None)

        grad_norm = np.linalg.norm(grad)
        max_grad = np.max(np.abs(grad))

        print("\n=== First-Order Optimality ===")
        print(f"Gradient norm: {grad_norm:.2e}")
        print(f"Max |gradient|: {max_grad:.2e}")

        assert max_grad < 1e-3, f"Gradient too large at MAP: max|grad| = {max_grad:.2e}"
        print("PASSED: First-order optimality satisfied")

    def test_predictions_reasonable(self, regression_data):
        """Verify predictions are non-negative and finite."""
        from pyneuroglm.regression.sklearn import BayesianGLMRegressor

        X, y = regression_data

        model = BayesianGLMRegressor(alpha=1.0, fit_intercept=True, initialize="lstsq")
        model.fit(X, y)

        y_pred = model.predict(X)

        assert np.all(np.isfinite(y_pred)), "Predictions should be finite"
        assert np.all(y_pred >= 0), "Poisson predictions should be non-negative"
        assert y_pred.shape == y.shape, "Prediction shape should match target"

        corr = np.corrcoef(y, y_pred)[0, 1]

        print("\n=== Prediction Quality ===")
        print(f"Predicted range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        print(f"Actual range: [{y.min():.0f}, {y.max():.0f}]")
        print(f"Correlation: {corr:.4f}")
        print("PASSED: Predictions are reasonable")

    def test_hessian_negative_definite(self, regression_data):
        """Verify Hessian is negative definite at MAP (second-order condition)."""
        from pyneuroglm.regression.sklearn import BayesianGLMRegressor
        from pyneuroglm.regression.posterior import poisson as poisson_posterior
        from pyneuroglm.regression.prior import ridge_Cinv
        from pyneuroglm.regression.nonlinearity import exp as exp_nlfun

        X, y = regression_data

        model = BayesianGLMRegressor(alpha=1.0, fit_intercept=True, initialize="lstsq")
        model.fit(X, y)

        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        w = np.concatenate([[model.intercept_], model.coef_])
        Cinv = ridge_Cinv(model.alpha, X_with_intercept.shape[1], intercept_prepended=True)

        _, _, H = poisson_posterior(w, X_with_intercept, y, Cinv, exp_nlfun, None)

        eigenvalues = np.linalg.eigvalsh(H)
        max_eigenvalue = np.max(eigenvalues)

        print("\n=== Second-Order Optimality ===")
        print(f"Max eigenvalue of Hessian: {max_eigenvalue:.2e}")
        print(f"Min eigenvalue of Hessian: {np.min(eigenvalues):.2e}")

        assert (
            max_eigenvalue < 0
        ), f"Hessian should be negative definite, max eigenvalue = {max_eigenvalue:.2e}"
        print("PASSED: Hessian is negative definite")
