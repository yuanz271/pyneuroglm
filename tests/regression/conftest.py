"""Shared fixtures and helpers for regression test suite."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from pyneuroglm import Experiment, Trial
from pyneuroglm.design import DesignMatrix
from pyneuroglm.regression.nonlinearity import exp as exp_nonlinearity


_MATLAB_DIR = Path(__file__).parent.parent.parent / "neuroGLM"
_EXAMPLE_DIR = Path(__file__).parent.parent.parent / "example"


@pytest.fixture
def matlab_dir():
    """Path to the MATLAB neuroGLM reference directory."""
    return _MATLAB_DIR


def _reference_poisson_negloglik(w, X, y):
    """
    Compute MATLAB-equivalent Poisson negative log-likelihood.

    Pure NumPy implementation of neuroGLM/matRegress/+glms/+neglog/poisson.m.
    Returns NEGATIVE log-likelihood, gradient, and Hessian.

    Parameters
    ----------
    w : ndarray of shape (p,)
        Weight vector.
    X : ndarray of shape (n, p)
        Design matrix.
    y : ndarray of shape (n,)
        Spike counts.

    Returns
    -------
    L : float
        Negative log-likelihood.
    dL : ndarray of shape (p,)
        Gradient of negative log-likelihood.
    H : ndarray of shape (p, p)
        Hessian of negative log-likelihood.
    """
    eta = X @ w
    f = np.exp(eta)
    nz = f > 0

    if np.any(y[~nz] != 0):
        return np.inf, np.full_like(w, np.nan), np.full((len(w), len(w)), np.nan)

    L = -np.sum(y[nz] * np.log(f[nz])) + np.sum(f)
    d = f[nz] - y[nz]
    dL = X[nz].T @ d
    h = f[nz]
    H = np.einsum("ij,i,ik->jk", X[nz], h, X[nz])

    return L, dL, H


@pytest.fixture
def reference_poisson_negloglik():
    """Return the MATLAB-equivalent Poisson negative log-likelihood function."""
    return _reference_poisson_negloglik


def _load_design_matrix_and_spikes():
    """
    Load the validated design matrix and binned spike counts.

    Returns
    -------
    X : ndarray of shape (n_bins, n_features)
        Design matrix (z-scored LFP columns, no bias).
    y : ndarray of shape (n_bins,)
        Binned spike counts.
    """
    dm_path = _MATLAB_DIR / "exampleDM.mat"
    if not dm_path.exists():
        dm_path = _EXAMPLE_DIR / "exampleDM.mat"

    mat = sio.loadmat(str(dm_path), squeeze_me=True, struct_as_record=False)
    X = mat["Z"]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    data_path = _MATLAB_DIR / "exampleData.mat"
    if not data_path.exists():
        data_path = _EXAMPLE_DIR / "exampleData.mat"

    mat = sio.loadmat(str(data_path), squeeze_me=True, struct_as_record=False)
    trials_raw = mat["trial"]

    expt = Experiment(time_unit="ms", binsize=1, eid="parity")
    expt.register_spike_train("sptrain", "Our Neuron")

    trial_indices = list(range(10))
    for i in trial_indices:
        t = trials_raw[i]
        trial = Trial(tid=i, duration=int(t.duration))
        trial["sptrain"] = np.asarray(t.sptrain, dtype=np.float64)
        expt.trials[i] = trial

    dm = DesignMatrix(expt)
    y = dm.get_binned_spike("sptrain", trial_indices)

    assert X.shape[0] == len(y)

    return X, y


@pytest.fixture
def matlab_regression_data():
    """Load design matrix and spike counts, skip if MATLAB fixtures missing."""
    dm_path = _MATLAB_DIR / "exampleDM.mat"
    data_path = _MATLAB_DIR / "exampleData.mat"
    if not dm_path.exists() or not data_path.exists():
        pytest.skip("MATLAB fixtures not found")
    return _load_design_matrix_and_spikes()
