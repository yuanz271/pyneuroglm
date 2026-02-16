"""Tests for pyneuroglm.regression.sklearn."""

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse

from pyneuroglm.regression.likelihood import poisson as poisson_loglik
from pyneuroglm.regression.nonlinearity import exp as exp_nlfun
from pyneuroglm.regression.posterior import poisson as poisson_posterior, initialize_lstsq, get_posterior_weights
from pyneuroglm.regression.prior import gaussian_zero_mean_inv, ridge_Cinv
from pyneuroglm.regression.sklearn import BayesianGLMRegressor, log_evidence_scorer


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_log_evidence_scorer_respects_fit_intercept():
    """log_evidence_scorer must not add intercept when fit_intercept=False."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    w_true = np.array([0.1, -0.2, 0.3])
    y = np.random.poisson(np.exp(X @ w_true))

    model_no = BayesianGLMRegressor(alpha=1.0, fit_intercept=False)
    model_no.fit(X, y)

    model_yes = BayesianGLMRegressor(alpha=1.0, fit_intercept=True)
    model_yes.fit(X, y)

    score_no = log_evidence_scorer(model_no, X, y)
    score_yes = log_evidence_scorer(model_yes, X, y)

    assert score_no != score_yes
    assert np.isfinite(score_no)
    assert np.isfinite(score_yes)


def test_initialize_zero_poisson_uses_log():
    """BayesianGLMRegressor with initialize='zero' should use log(mean(y)) for Poisson."""
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = np.random.poisson(5.0, size=200)

    model = BayesianGLMRegressor(alpha=1.0, initialize="zero")
    model.fit(X, y)

    np.testing.assert_allclose(np.exp(model.intercept_), np.mean(y), rtol=0.1)


# ---------------------------------------------------------------------------
# MATLAB parity tests
# ---------------------------------------------------------------------------


def _to_dense(x):
    """Convert sparse matrix to dense, pass through dense arrays."""
    return x.toarray() if scipy.sparse.issparse(x) else x


class TestMAPRegression:
    """MAP regression end-to-end tests."""

    @pytest.fixture
    def data(self, matlab_regression_data):
        return matlab_regression_data

    def test_convergence(self, data):
        """Verify BayesianGLMRegressor converges on validated data."""
        X, y = data

        model = BayesianGLMRegressor(alpha=1.0, fit_intercept=True, initialize="lstsq")
        model.fit(X, y)

        assert len(model.coef_) == X.shape[1]

    def test_first_order_optimality(self, data):
        """Verify gradient is near zero at MAP solution."""
        X, y = data

        model = BayesianGLMRegressor(alpha=1.0, fit_intercept=True, initialize="lstsq")
        model.fit(X, y)

        X_ = np.column_stack((np.ones(X.shape[0]), X))
        w = np.concatenate([[model.intercept_], model.coef_])
        Cinv = ridge_Cinv(model.alpha, X_.shape[1], intercept_prepended=True)

        _, grad, _ = poisson_posterior(w, X_, y, Cinv, exp_nlfun, None)

        assert np.max(np.abs(grad)) < 1e-3

    def test_predictions_reasonable(self, data):
        """Verify predictions are non-negative and finite."""
        X, y = data

        model = BayesianGLMRegressor(alpha=1.0, fit_intercept=True, initialize="lstsq")
        model.fit(X, y)

        y_pred = model.predict(X)

        assert np.all(np.isfinite(y_pred))
        assert np.all(y_pred >= 0)
        assert y_pred.shape == y.shape

    def test_hessian_negative_definite(self, data):
        """Verify Hessian is negative definite at MAP."""
        X, y = data

        model = BayesianGLMRegressor(alpha=1.0, fit_intercept=True, initialize="lstsq")
        model.fit(X, y)

        X_ = np.column_stack((np.ones(X.shape[0]), X))
        w = np.concatenate([[model.intercept_], model.coef_])
        Cinv = ridge_Cinv(model.alpha, X_.shape[1], intercept_prepended=True)

        _, _, H = poisson_posterior(w, X_, y, Cinv, exp_nlfun, None)

        assert np.max(np.linalg.eigvalsh(H)) < 0

    def test_map_parity_with_matlab(self, matlab_dir):
        """Compare Python MAP objective and weights against MATLAB exampleMAP.mat."""
        map_path = matlab_dir / "exampleMAP.mat"
        if not map_path.exists():
            pytest.skip("MATLAB MAP fixture not found")

        mat = sio.loadmat(map_path)

        X = _to_dense(mat["X"])
        y = _to_dense(mat["y"]).ravel()
        alpha = float(_to_dense(mat["alpha"]).ravel()[0])
        wml_matlab = _to_dense(mat["wml"]).ravel()
        nlogli_matlab = float(_to_dense(mat["nlogli"]).ravel()[0])

        Cinv = ridge_Cinv(alpha, X.shape[1], intercept_prepended=True)
        w_py, _, _ = get_posterior_weights(X, y, Cinv, dist="poisson", initialize=initialize_lstsq)

        inds = np.arange(len(y))
        L_py, _, _ = poisson_loglik(w_py, X, y, exp_nlfun, inds)
        P_py, _, _ = gaussian_zero_mean_inv(w_py, Cinv)
        obj_py = -(L_py + P_py)

        assert abs(obj_py - nlogli_matlab) < 0.01
        assert np.corrcoef(w_py, wml_matlab)[0, 1] > 0.9999

    def test_initialization_parity_with_matlab(self, matlab_dir):
        """Verify initialize_lstsq matches MATLAB's (X'X + Cinv) \\ (X'y)."""
        map_path = matlab_dir / "exampleMAP.mat"
        if not map_path.exists():
            pytest.skip("MATLAB MAP fixture not found")

        mat = sio.loadmat(map_path)

        X = _to_dense(mat["X"])
        y = _to_dense(mat["y"]).ravel()
        alpha = float(_to_dense(mat["alpha"]).ravel()[0])
        w0_matlab = _to_dense(mat["w0"]).ravel()

        Cinv = ridge_Cinv(alpha, X.shape[1], intercept_prepended=True)
        w0_python = initialize_lstsq(X, y, Cinv)

        assert np.allclose(w0_python, w0_matlab, atol=1e-10)
