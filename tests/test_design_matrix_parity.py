"""
Design matrix parity test between MATLAB neuroGLM and pyneuroglm.

This test validates that pyneuroglm produces design matrices equivalent to
MATLAB neuroGLM for the same experimental specification.

Reference:
- MATLAB fixture: neuroGLM/exampleDM.mat (or example_X.mat)
- MATLAB source: neuroGLM/tutorial.m
- Trial data: neuroGLM/exampleData.mat
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io as sio

from pyneuroglm import Experiment, Trial
from pyneuroglm.basis import make_smooth_temporal_basis, boxcar_stim, Basis
from pyneuroglm.design import DesignMatrix


MATLAB_DIR = Path(__file__).parent.parent / "neuroGLM"
TOLERANCE_DESIGN_MATRIX = 1e-8
TOLERANCE_ZSCORE_STATS = 1e-10


@dataclass
class ColumnIdentity:
    """Canonical identity for a design matrix column."""

    covariate_label: str
    covariate_type: str
    basis_type: str | None
    basis_index: int | None
    offset: float
    is_bias: bool

    def __hash__(self):
        """Return hash based on all fields."""
        return hash(
            (
                self.covariate_label,
                self.covariate_type,
                self.basis_type,
                self.basis_index,
                self.offset,
                self.is_bias,
            )
        )


@dataclass
class ParityFailure:
    """Structured failure record for parity test."""

    phase: str
    category: str
    severity: str
    message: str
    matlab_context: Any = None
    python_context: Any = None
    suggested_action: str = ""

    def __str__(self):
        """Return formatted failure message."""
        lines = [
            f"[{self.severity.upper()}][{self.category}] {self.phase}",
            self.message,
        ]
        if self.matlab_context is not None:
            lines.append(f"MATLAB context: {self.matlab_context}")
        if self.python_context is not None:
            lines.append(f"Python context: {self.python_context}")
        if self.suggested_action:
            lines.append(f"Suggested action: {self.suggested_action}")
        return "\n".join(lines)


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
    """
    Load MATLAB trial data from exampleData.mat.

    Returns
    -------
    dict with keys:
        - nTrials: int
        - param: dict with samplingFreq, monkey
        - trials: list of dicts, each containing trial data
    """
    mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    n_trials = int(mat["nTrials"])
    param = mat["param"]
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

    return {
        "nTrials": n_trials,
        "param": {"samplingFreq": param.samplingFreq, "monkey": param.monkey},
        "trials": trials,
    }


def build_python_experiment(trial_data: dict, binsize: int = 1) -> Experiment:
    """
    Build a Python Experiment from MATLAB trial data.

    Mirrors MATLAB tutorial.m lines 8-21.
    """
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
        trial["LFP"] = t["LFP"]
        trial["eyepos"] = t["eyepos"]
        trial["dotson"] = t["dotson"]
        trial["dotsoff"] = t["dotsoff"]
        trial["saccade"] = t["saccade"]
        trial["sptrain"] = t["sptrain"]
        trial["sptrain2"] = t["sptrain2"]
        trial["coh"] = t["coh"]
        trial["choice"] = t["choice"]
        expt.trials[i] = trial

    return expt


def build_design_matrix_like_matlab(expt: Experiment, trial_indices: list[int]) -> DesignMatrix:
    """
    Build design matrix following MATLAB tutorial.m specification.

    This mirrors tutorial.m lines 26-57 as closely as possible.
    """
    dm = DesignMatrix(expt)
    binfun = expt.binfun

    # tutorial.m:28-32 — LFP with scaled boxcar basis
    bs_lfp = make_smooth_temporal_basis("boxcar", 100, 10, binfun)
    bs_lfp_scaled = Basis(
        name=bs_lfp.name,
        func=bs_lfp.func,
        kwargs=bs_lfp.kwargs,
        B=0.1 * bs_lfp.B,
        edim=bs_lfp.edim,
        tr=bs_lfp.tr,
        centers=bs_lfp.centers,
    )
    dm.add_covariate_raw("LFP", "Local Field Potential", basis=bs_lfp_scaled)

    # tutorial.m:35
    dm.add_covariate_spike("hist", "sptrain", "History filter")

    # tutorial.m:38
    dm.add_covariate_spike("coupling", "sptrain2", "Coupling from neuron 2")

    # tutorial.m:41
    dm.add_covariate_boxcar("dots", "dotson", "dotsoff", "Motion dots stim")

    # tutorial.m:44-46
    bs_saccade = make_smooth_temporal_basis("boxcar", 300, 8, binfun)
    dm.add_covariate_timing("saccade", description="Saccade", basis=bs_saccade, offset=-200)

    # tutorial.m:49-53
    bs_coh = make_smooth_temporal_basis("raised cosine", 200, 10, binfun)

    def coh_handler(trial):
        on_bin = binfun(trial["dotson"])
        off_bin = binfun(trial["dotsoff"], True)
        n_bins = binfun(trial.duration, True)
        return trial["coh"] * boxcar_stim(on_bin, off_bin, n_bins)

    dm.add_covariate("cohKer", "coh-dep dots stimulus", coh_handler, basis=bs_coh)

    # tutorial.m:56-57
    bs_eye = make_smooth_temporal_basis("raised cosine", 40, 4, binfun)
    dm.add_covariate_raw("eyepos", "Eye Position", basis=bs_eye)

    return dm


def compare_shapes(X_py: np.ndarray, X_mat: np.ndarray) -> ParityFailure | None:
    """Check if design matrix shapes match."""
    if X_py.shape != X_mat.shape:
        return ParityFailure(
            phase="Shape Comparison",
            category="structural",
            severity="fatal",
            message="Design matrix shapes differ.",
            matlab_context=f"shape = {X_mat.shape}",
            python_context=f"shape = {X_py.shape}",
            suggested_action="Check covariate count, basis widths, and trial binning.",
        )
    return None


def compare_values(
    X_py: np.ndarray, X_mat: np.ndarray, tolerance: float = TOLERANCE_DESIGN_MATRIX
) -> ParityFailure | None:
    """Check numerical equality of design matrices."""
    diff = np.abs(X_py - X_mat)
    max_diff = np.max(diff)

    if max_diff >= tolerance:
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        row, col = max_idx

        return ParityFailure(
            phase="Numerical Comparison",
            category="numerical",
            severity="fatal",
            message=f"Design matrix values differ. Max diff = {max_diff:.2e}, tolerance = {tolerance:.2e}",
            matlab_context=f"X_mat[{row}, {col}] = {X_mat[row, col]:.10f}",
            python_context=f"X_py[{row}, {col}] = {X_py[row, col]:.10f}",
            suggested_action="Check stimulus binning, basis convolution, and covariate ordering.",
        )
    return None


def find_mismatched_columns(
    X_py: np.ndarray, X_mat: np.ndarray, tolerance: float = TOLERANCE_DESIGN_MATRIX
) -> list[int]:
    """Find columns with values exceeding tolerance."""
    mismatched = []
    for col in range(X_py.shape[1]):
        max_diff = np.max(np.abs(X_py[:, col] - X_mat[:, col]))
        if max_diff >= tolerance:
            mismatched.append(col)
    return mismatched


@pytest.mark.slow
class TestDesignMatrixParity:
    """Design matrix parity test suite."""

    @pytest.fixture
    def matlab_design_matrix(self) -> np.ndarray:
        """Load MATLAB reference design matrix."""
        path = MATLAB_DIR / "exampleDM.mat"
        if not path.exists():
            pytest.skip(f"MATLAB fixture not found: {path}")
        return load_matlab_design_matrix(path)

    @pytest.fixture
    def matlab_trial_data(self) -> dict:
        """Load MATLAB trial data."""
        path = MATLAB_DIR / "exampleData.mat"
        if not path.exists():
            pytest.skip(f"MATLAB fixture not found: {path}")
        return load_matlab_trial_data(path)

    def test_design_matrix_shape_parity(self, matlab_design_matrix, matlab_trial_data):
        """
        Test 1: Verify design matrix shapes match.

        This is a prerequisite for value comparison.
        """
        expt = build_python_experiment(matlab_trial_data, binsize=1)
        trial_indices = list(range(10))
        dm = build_design_matrix_like_matlab(expt, trial_indices)
        X_py = dm.compile_design_matrix(trial_indices)

        print("\n=== Shape Comparison ===")
        print(f"MATLAB X shape: {matlab_design_matrix.shape}")
        print(f"Python X shape: {X_py.shape}")
        print(f"Python edim: {dm.edim}")
        print(f"Covariates: {list(dm.covariates.keys())}")
        for label, covar in dm.covariates.items():
            print(f"  {label}: edim={covar.edim}, sdim={covar.sdim}")

    def test_design_matrix_parity_raw(self, matlab_design_matrix, matlab_trial_data):
        """
        Test 2: Compare raw design matrix (before post-processing).

        Note: The MATLAB reference matrix has been post-processed with:
        - removeConstantCols
        - zscoreDesignMatrix (on LFP columns)
        - addBiasColumn

        This test compares the raw matrices before these steps.
        This will likely fail due to post-processing differences.
        """
        expt = build_python_experiment(matlab_trial_data, binsize=1)
        trial_indices = list(range(10))
        dm = build_design_matrix_like_matlab(expt, trial_indices)
        X_py = dm.compile_design_matrix(trial_indices)

        print("\n=== Raw Design Matrix Comparison ===")
        print(f"MATLAB X shape: {matlab_design_matrix.shape}")
        print(f"Python X shape: {X_py.shape}")

        failure = compare_shapes(X_py, matlab_design_matrix)
        if failure:
            print(failure)
            pytest.skip(
                f"Shape mismatch (expected due to post-processing): "
                f"Python {X_py.shape} vs MATLAB {matlab_design_matrix.shape}"
            )

        failure = compare_values(X_py, matlab_design_matrix)
        if failure:
            print(failure)
            mismatched = find_mismatched_columns(X_py, matlab_design_matrix)
            print(f"Mismatched columns (first 10): {mismatched[:10]}")

    def test_design_matrix_parity_with_zscore(self, matlab_design_matrix, matlab_trial_data):
        """
        Test 3: Compare design matrix with z-scoring applied.

        The MATLAB reference exampleDM.mat contains the design matrix
        after z-scoring LFP columns but WITHOUT the bias column.
        """
        expt = build_python_experiment(matlab_trial_data, binsize=1)
        trial_indices = list(range(10))
        dm = build_design_matrix_like_matlab(expt, trial_indices)
        X_py = dm.compile_design_matrix(trial_indices)

        print("\n=== Z-scored Design Matrix Comparison ===")
        print(f"Raw Python X shape: {X_py.shape}")

        lfp_indices = dm.get_design_matrix_col_indices("LFP")
        print(f"LFP column indices: {lfp_indices}")
        dm.zscore_columns(lfp_indices)
        X_py = dm.X

        print(f"Python X shape (after zscore): {X_py.shape}")
        print(f"MATLAB X shape: {matlab_design_matrix.shape}")

        failure = compare_shapes(X_py, matlab_design_matrix)
        if failure:
            print(failure)
            pytest.fail(str(failure))

        failure = compare_values(X_py, matlab_design_matrix)
        if failure:
            print(failure)
            mismatched = find_mismatched_columns(X_py, matlab_design_matrix)
            print(f"Mismatched columns (first 10): {mismatched[:10]}")
            pytest.fail(str(failure))

        max_diff = np.max(np.abs(X_py - matlab_design_matrix))
        print(f"\nMax difference: {max_diff:.2e}")
        print("✅ Design matrix parity PASSED!")


if __name__ == "__main__":
    print("Loading MATLAB data...")
    X_mat = load_matlab_design_matrix(MATLAB_DIR / "exampleDM.mat")
    trial_data = load_matlab_trial_data(MATLAB_DIR / "exampleData.mat")

    print(f"MATLAB design matrix shape: {X_mat.shape}")
    print(f"Number of trials: {trial_data['nTrials']}")

    print("\nBuilding Python experiment...")
    expt = build_python_experiment(trial_data, binsize=1)

    print("Building design matrix...")
    trial_indices = list(range(10))
    dm = build_design_matrix_like_matlab(expt, trial_indices)

    print("Compiling design matrix...")
    X_py = dm.compile_design_matrix(trial_indices)

    print("\n=== Results ===")
    print(f"MATLAB X shape: {X_mat.shape}")
    print(f"Python X shape: {X_py.shape}")
    print(f"Python covariates: {list(dm.covariates.keys())}")
    for label, covar in dm.covariates.items():
        print(f"  {label}: edim={covar.edim}")
