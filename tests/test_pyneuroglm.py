"""Smoke tests covering the high-level pyneuroglm API."""

from collections import namedtuple
from pathlib import Path

import numpy as np

from pyneuroglm.basis import make_smooth_temporal_basis, conv_basis, make_nonlinear_raised_cos
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
