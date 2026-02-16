"""Tests for pyneuroglm.basis."""

from pathlib import Path

import numpy as np

from pyneuroglm.basis import (
    make_smooth_temporal_basis,
    conv_basis,
    make_nonlinear_raised_cos,
    delta_stim,
)
from pyneuroglm.experiment import Experiment


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
    x = np.random.randn(100, 2)
    X = conv_basis(x, B, offset=5)
    assert X.shape == (100, 10)
    X = conv_basis(x, B, offset=-5)
    assert X.shape == (100, 10)


def test_make_nonlinear_raised_cos():
    """Verify nonlinear raised cosine basis matches MATLAB reference data."""
    basis_matlab = np.load(Path(__file__).parent / "basis.npy", allow_pickle=True)
    B = basis_matlab["B"][()]
    param = basis_matlab["param"][()]
    n_bases, binsize, end_points, nl_offset = param.tolist()

    basis = make_nonlinear_raised_cos(n_bases, binsize, end_points, nl_offset)

    basis_recons = basis.func(**basis.kwargs)
    assert np.all(basis.B == basis_recons.B)
    assert np.allclose(basis.B[:-1], B)  # NOTE: Unequal size


def test_delta_stim_filters_negative_indices():
    """delta_stim must silently ignore negative bin indices."""
    bt = np.array([-2, -1, 0, 3, 5, 10])
    n_bins = 8
    stim = delta_stim(bt, n_bins)

    assert stim.shape == (n_bins, 1)
    expected = np.zeros((n_bins, 1))
    expected[0] = 1.0
    expected[3] = 1.0
    expected[5] = 1.0
    np.testing.assert_array_equal(stim, expected)
