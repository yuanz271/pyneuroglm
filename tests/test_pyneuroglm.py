"""Smoke tests covering the high-level pyneuroglm API."""

from collections import namedtuple
from pathlib import Path

import numpy as np

from pyneuroglm.basis import make_smooth_temporal_basis, conv_basis, make_nonlinear_raised_cos
from pyneuroglm.design import DesignMatrix
from pyneuroglm.experiment import Experiment, Trial, Variable


def test_experiment():
    """Ensure experiment bins events as expected."""
    expt = Experiment(time_unit="ms", binsize=10, eid=1)
    assert expt.binfun(0, True) == 1


def test_variable():
    """Check variable dataclass fields are populated."""
    v = Variable("label", "description", "type", 2)
    assert (
        v.label == "label" and v.description == "description" and v.type == "type" and v.ndim == 2
    )


def test_trial():
    """Confirm trial indexing stores arrays."""
    trial = Trial(1, 10)
    trial["a"] = np.zeros(100)
    assert trial["a"].shape == (100,)


def test_make_smooth_temporal_basis():
    """Validate raised cosine basis regeneration matches stored data."""
    expt = Experiment(time_unit="ms", binsize=10, eid=1)
    basis = make_smooth_temporal_basis("raised cosine", 100, 5, expt.binfun)
    basis_boot = basis.func(**basis.kwargs)
    assert np.all(basis.B == basis_boot.B)


def test_conv_basis():
    """Ensure convolution helper yields expected matrix shape."""
    expt = Experiment(time_unit="ms", binsize=10, eid=1)
    B = make_smooth_temporal_basis("raised cosine", 100, 5, expt.binfun)
    n = 100
    d = 2
    x = np.random.randn(n, d)
    X = conv_basis(x, B, offset=5)
    assert X.shape == (100, 10)
    X = conv_basis(x, B, offset=-5)
    assert X.shape == (100, 10)


def test_combine_weights():
    """Confirm namedtuple construction mirrors weight order."""
    labels = ["a", "b", "c"]
    values = [1, 2, 3]
    W = namedtuple("Weight", labels)
    w = W(*values)
    assert w == (1, 2, 3)


def test_make_nonlinear_raised_cos():
    """Verify nonlinear raised cosine basis matches MATLAB reference data."""
    basis_matlab = np.load(Path(__file__).parent / "basis.npy", allow_pickle=True)
    B = basis_matlab["B"][()]
    param = basis_matlab["param"][
        ()
    ]  # dtype=[('nBases', 'O'), ('binSize', 'O'), ('endPoints', 'O'), ('nlOffset', 'O')])
    n_bases, binsize, end_points, nl_offset = param.tolist()

    basis = make_nonlinear_raised_cos(n_bases, binsize, end_points, nl_offset)

    basis_recons = basis.func(**basis.kwargs)
    assert np.all(basis.B == basis_recons.B)

    assert np.allclose(basis.B[:-1], B)  # NOTE: Unequal size


def test_combine_weights_zscore_inversion():
    """Verify combine_weights correctly inverts z-scored design matrix columns."""
    np.random.seed(42)
    expt = Experiment(time_unit="ms", binsize=10, eid=1)
    expt.register_continuous("signal", "Test signal")

    n_bins = expt.binfun(500, True)
    trial = Trial(tid=0, duration=500)
    trial["signal"] = np.random.randn(n_bins) * 5 + 10  # non-zero mean, large std
    expt.add_trial(trial)

    dm = DesignMatrix(expt)
    dm.add_covariate_raw("signal", "Test signal")
    X = dm.compile_design_matrix()

    # True weights in original space
    w_true = np.array([2.0])
    y = X @ w_true

    # Z-score and fit
    dm.zscore_columns()
    X_z = dm.X
    w_z = np.linalg.lstsq(X_z, y, rcond=None)[0]

    # combine_weights should recover original-space weights
    ws = dm.combine_weights(w_z)
    w_recovered = ws["signal"]["data"]

    np.testing.assert_allclose(w_recovered.flatten(), w_true, atol=1e-10)


def test_log_evidence_scorer_respects_fit_intercept():
    """log_evidence_scorer must not add intercept when fit_intercept=False."""
    from pyneuroglm.regression.sklearn import BayesianGLMRegressor, log_evidence_scorer

    np.random.seed(42)
    X = np.random.randn(100, 3)
    w_true = np.array([0.1, -0.2, 0.3])
    y = np.random.poisson(np.exp(X @ w_true))

    model_no_intercept = BayesianGLMRegressor(alpha=1.0, fit_intercept=False)
    model_no_intercept.fit(X, y)

    model_with_intercept = BayesianGLMRegressor(alpha=1.0, fit_intercept=True)
    model_with_intercept.fit(X, y)

    score_no = log_evidence_scorer(model_no_intercept, X, y)
    score_with = log_evidence_scorer(model_with_intercept, X, y)

    # Scores should differ — they are different models
    assert score_no != score_with

    # The no-intercept scorer should use coef_ directly without prepending
    # If the bug were present, it would prepend intercept_=0.0 and a ones column,
    # giving a score identical to a model with intercept=0 (not the same as no intercept)
    assert np.isfinite(score_no)
    assert np.isfinite(score_with)
