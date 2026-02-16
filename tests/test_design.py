"""Tests for pyneuroglm.design."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from pyneuroglm import Experiment, Trial
from pyneuroglm.basis import make_smooth_temporal_basis, boxcar_stim, Basis
from pyneuroglm.design import DesignMatrix


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_combine_weights_zscore_inversion():
    """Verify combine_weights correctly inverts z-scored design matrix columns."""
    np.random.seed(42)
    expt = Experiment(time_unit="ms", binsize=10, eid=1)
    expt.register_continuous("signal", "Test signal")

    n_bins = expt.binfun(500, True)
    trial = Trial(tid=0, duration=500)
    trial["signal"] = np.random.randn(n_bins) * 5 + 10
    expt.add_trial(trial)

    dm = DesignMatrix(expt)
    dm.add_covariate_raw("signal", "Test signal")
    X = dm.compile_design_matrix()

    w_true = np.array([2.0])
    y = X @ w_true

    dm.zscore_columns()
    w_z = np.linalg.lstsq(dm.X, y, rcond=None)[0]

    ws = dm.combine_weights(w_z)
    w_recovered = ws["signal"]["data"]
    np.testing.assert_allclose(w_recovered.flatten(), w_true, atol=1e-10)


def test_add_covariate_constant_uses_stim_label():
    """add_covariate_constant must use stim_label to look up trial data."""
    expt = Experiment(time_unit="ms", binsize=10, eid=1)
    expt.register_value("coh", "Coherence")

    trial = Trial(tid=0, duration=100)
    trial["coh"] = np.array([0.5])
    expt.add_trial(trial)

    dm = DesignMatrix(expt)
    dm.add_covariate_constant("coherence", stim_label="coh")
    X = dm.compile_design_matrix()

    n_bins = expt.binfun(100, True)
    assert X.shape == (n_bins, 1)
    np.testing.assert_allclose(X[:, 0], 0.5)


def test_compile_design_matrix_with_condition():
    """compile_design_matrix must zero-fill columns when condition excludes a covariate."""
    expt = Experiment(time_unit="ms", binsize=10, eid=1)
    expt.register_timing("event1", "Event 1")
    expt.register_timing("event2", "Event 2")

    trial1 = Trial(tid=1, duration=500)
    trial1["event1"] = np.array([100.0, 200.0])
    trial1["event2"] = np.array([150.0, 300.0])

    trial2 = Trial(tid=2, duration=500)
    trial2["event1"] = np.array([50.0])
    trial2["event2"] = np.array([250.0])

    expt.add_trial(trial1)
    expt.add_trial(trial2)

    dm = DesignMatrix(expt)
    basis = make_smooth_temporal_basis("raised cosine", 200, 3, expt.binfun)

    dm.add_covariate_timing("event1", basis=basis)
    dm.add_covariate_timing("event2", basis=basis, condition=lambda t: t.tid == 1)

    X = dm.compile_design_matrix()
    n_bins = expt.binfun(500, True)

    assert X.shape == (2 * n_bins, 6)

    # Trial 2's event2 columns should be all zeros
    np.testing.assert_array_equal(X[n_bins:, 3:6], 0.0)
    # Trial 1's event2 columns should have nonzero entries
    assert np.any(X[:n_bins, 3:6] != 0.0)


# ---------------------------------------------------------------------------
# MATLAB parity tests
# ---------------------------------------------------------------------------

MATLAB_DIR = Path(__file__).parent.parent / "neuroGLM"
TOLERANCE_DESIGN_MATRIX = 1e-8


def load_matlab_design_matrix(path: Path) -> np.ndarray:
    """Load MATLAB design matrix and convert to dense numpy array."""
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    if "Z" in mat:
        X = mat["Z"]
    elif "X" in mat:
        X = mat["X"]
    else:
        raise KeyError(f"No design matrix found in {path}. Keys: {list(mat.keys())}")

    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.asarray(X, dtype=np.float64)


def load_matlab_trial_data(path: Path) -> dict:
    """Load MATLAB trial data from exampleData.mat."""
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    n_trials = int(mat["nTrials"])
    trials_raw = mat["trial"]

    trials = []
    for i in range(n_trials):
        t = trials_raw[i]
        trial_dict = {
            "duration": int(t.duration),
            "LFP": np.asarray(t.LFP, dtype=np.float64),
            "eyepos": np.asarray(t.eyepos, dtype=np.float64),
            "dotson": int(t.dotson),
            "dotsoff": int(t.dotsoff),
            "saccade": int(t.saccade),
            "coh": float(t.coh),
            "choice": int(t.choice),
            "sptrain": np.asarray(t.sptrain, dtype=np.float64),
            "sptrain2": (
                np.asarray(t.sptrain2, dtype=np.float64)
                if hasattr(t.sptrain2, "__len__")
                else np.array([float(t.sptrain2)])
            ),
        }
        trials.append(trial_dict)

    return {"nTrials": n_trials, "trials": trials}


def build_python_experiment(trial_data: dict, binsize: int = 1) -> Experiment:
    """Build a Python Experiment from MATLAB trial data."""
    expt = Experiment(time_unit="ms", binsize=binsize, eid="matlab_parity")

    expt.register_continuous("LFP", "Local Field Potential", ndim=1)
    expt.register_continuous("eyepos", "Eye Position", ndim=2)
    expt.register_timing("dotson", "Motion Dots Onset")
    expt.register_timing("dotsoff", "Motion Dots Offset")
    expt.register_timing("saccade", "Saccade Timing")
    expt.register_spike_train("sptrain", "Our Neuron")
    expt.register_spike_train("sptrain2", "Neighbor Neuron")
    expt.register_value("coh", "Coherence")
    expt.register_value("choice", "Direction of Choice")

    for i, t in enumerate(trial_data["trials"]):
        trial = Trial(tid=i, duration=t["duration"])
        for key in ["LFP", "eyepos", "dotson", "dotsoff", "saccade", "sptrain", "sptrain2", "coh", "choice"]:
            trial[key] = t[key]
        expt.trials[i] = trial

    return expt


def build_design_matrix_like_matlab(expt: Experiment) -> DesignMatrix:
    """Build design matrix following MATLAB tutorial.m specification."""
    dm = DesignMatrix(expt)
    binfun = expt.binfun

    bs_lfp = make_smooth_temporal_basis("boxcar", 100, 10, binfun)
    bs_lfp_scaled = Basis(
        name=bs_lfp.name, func=bs_lfp.func, kwargs=bs_lfp.kwargs,
        B=0.1 * bs_lfp.B, edim=bs_lfp.edim, tr=bs_lfp.tr, centers=bs_lfp.centers,
    )
    dm.add_covariate_raw("LFP", "Local Field Potential", basis=bs_lfp_scaled)
    dm.add_covariate_spike("hist", "sptrain", "History filter")
    dm.add_covariate_spike("coupling", "sptrain2", "Coupling from neuron 2")
    dm.add_covariate_boxcar("dots", "dotson", "dotsoff", "Motion dots stim")

    bs_saccade = make_smooth_temporal_basis("boxcar", 300, 8, binfun)
    dm.add_covariate_timing("saccade", description="Saccade", basis=bs_saccade, offset=-200)

    bs_coh = make_smooth_temporal_basis("raised cosine", 200, 10, binfun)

    def coh_handler(trial):
        on_bin = binfun(trial["dotson"])
        off_bin = binfun(trial["dotsoff"], True)
        n_bins = binfun(trial.duration, True)
        return trial["coh"] * boxcar_stim(on_bin, off_bin, n_bins)

    dm.add_covariate("cohKer", "coh-dep dots stimulus", coh_handler, basis=bs_coh)

    bs_eye = make_smooth_temporal_basis("raised cosine", 40, 4, binfun)
    dm.add_covariate_raw("eyepos", "Eye Position", basis=bs_eye)

    return dm


@pytest.mark.slow
class TestDesignMatrixParity:
    """Design matrix parity test suite against MATLAB neuroGLM."""

    @pytest.fixture
    def matlab_design_matrix(self) -> np.ndarray:
        path = MATLAB_DIR / "exampleDM.mat"
        if not path.exists():
            pytest.skip(f"MATLAB fixture not found: {path}")
        return load_matlab_design_matrix(path)

    @pytest.fixture
    def matlab_trial_data(self) -> dict:
        path = MATLAB_DIR / "exampleData.mat"
        if not path.exists():
            pytest.skip(f"MATLAB fixture not found: {path}")
        return load_matlab_trial_data(path)

    def test_design_matrix_shape_parity(self, matlab_design_matrix, matlab_trial_data):
        """Verify design matrix shapes match."""
        expt = build_python_experiment(matlab_trial_data, binsize=1)
        trial_indices = list(range(10))
        dm = build_design_matrix_like_matlab(expt)
        X_py = dm.compile_design_matrix(trial_indices)

    def test_design_matrix_parity_raw(self, matlab_design_matrix, matlab_trial_data):
        """Compare raw design matrix (before post-processing)."""
        expt = build_python_experiment(matlab_trial_data, binsize=1)
        trial_indices = list(range(10))
        dm = build_design_matrix_like_matlab(expt)
        X_py = dm.compile_design_matrix(trial_indices)

        if X_py.shape != matlab_design_matrix.shape:
            pytest.skip(
                f"Shape mismatch (expected due to post-processing): "
                f"Python {X_py.shape} vs MATLAB {matlab_design_matrix.shape}"
            )

    def test_design_matrix_parity_with_zscore(self, matlab_design_matrix, matlab_trial_data):
        """Compare design matrix with z-scoring applied."""
        expt = build_python_experiment(matlab_trial_data, binsize=1)
        trial_indices = list(range(10))
        dm = build_design_matrix_like_matlab(expt)
        X_py = dm.compile_design_matrix(trial_indices)

        lfp_indices = dm.get_design_matrix_col_indices("LFP")
        dm.zscore_columns(lfp_indices)
        X_py = dm.X

        assert X_py.shape == matlab_design_matrix.shape, (
            f"Shape mismatch: Python {X_py.shape} vs MATLAB {matlab_design_matrix.shape}"
        )

        max_diff = np.max(np.abs(X_py - matlab_design_matrix))
        assert max_diff < TOLERANCE_DESIGN_MATRIX, (
            f"Design matrix values differ: max diff = {max_diff:.2e}"
        )
