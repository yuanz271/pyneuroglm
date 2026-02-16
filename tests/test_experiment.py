"""Tests for pyneuroglm.experiment."""

import numpy as np

from pyneuroglm.experiment import Experiment, Trial, Variable


def test_experiment_binfun():
    """Ensure experiment bins events as expected."""
    expt = Experiment(time_unit="ms", binsize=10, eid=1)
    assert expt.binfun(0, True) == 1


def test_variable():
    """Check variable dataclass fields are populated."""
    v = Variable("label", "description", "type", 2)
    assert v.label == "label"
    assert v.description == "description"
    assert v.type == "type"
    assert v.ndim == 2


def test_trial():
    """Confirm trial indexing stores arrays."""
    trial = Trial(1, 10)
    trial["a"] = np.zeros(100)
    assert trial["a"].shape == (100,)
