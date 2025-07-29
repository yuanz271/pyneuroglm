from math import ceil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .experiment import Experiment
from .basis import conv_basis, delta_stim, boxcar_stim, make_nonlinear_raised_cos, Basis
from .util import zscore


__all__ = ["DesignMatrix", "Covariate"]


@dataclass
class DesignMatrix:
    """
    GLM design matrix builder for experiments.

    :param experiment: The Experiment object containing trial data and binning functions.
    :type experiment: Experiment
    :param covariates: Dictionary of covariate label to Covariate objects.
    :type covariates: dict
    :param bias: Whether to include a bias (constant) column in the design matrix.
    :type bias: bool
    """
    experiment: Experiment
    covariates: dict = field(default_factory=dict)
    bias: bool = False
    zstats: dict = field(default_factory=dict)
    _X: NDArray | None = field(init=False, default=None)
    
    @property
    def X(self) -> NDArray:
        if self._X is None:
            raise RuntimeError("Design matrix has not been compiled")
        return self._X

    @property
    def edim(self):
        """
        Total effective dimension (number of columns) of the design matrix.

        :returns: Sum of effective dimensions of all covariates.
        :rtype: int
        """
        return sum((covar.edim for covar in self.covariates.values()))

    def prepend_constant(self, bias=True):
        """
        Add a constant (bias) column to the design matrix.

        :param bias: Whether to include the bias column.
        :type bias: bool
        """
        self.bias = bias

    def add_covariate(
        self,
        label,
        description,
        handler,
        basis=None,
        offset=0,
        condition: Callable | None = None,
    ):
        """
        Add a covariate to the design.

        :param label: Covariate label.
        :type label: str
        :param description: Description of the covariate.
        :type description: str
        :param handler: Function to generate the stimulus array for each trial.
        :type handler: Callable
        :param basis: Basis object for the covariate, or None for raw.
        :type basis: Basis or None
        :param offset: Offset to apply to the covariate (in time units).
        :type offset: float
        :param condition: Optional function to filter trials for this covariate.
        :type condition: Callable or None
        """
        self.covariates[label] = Covariate(
            self, label, description, handler, basis, offset, condition
        )

    def add_covariate_constant(self, label, stim_label=None, description=None, **kwargs):
        binfun = self.experiment.binfun
        if stim_label is None:
            stim_label = label

        self.covariates[label] = Covariate(
            self,
            label,
            description,
            constant_stim(label, binfun),
            **kwargs,
        )

    def add_covariate_timing(self, label, stim_label=None, description=None, **kwargs):
        """
        Add a covariate based on event timing (delta function).

        :param label: Covariate label.
        :type label: str
        :param stim_label: Label in trial dict for event times (defaults to label).
        :type stim_label: str or None
        :param description: Description of the covariate.
        :type description: str or None
        :param kwargs: Additional arguments for Covariate.
        """
        binfun = self.experiment.binfun
        if stim_label is None:
            stim_label = label

        self.covariates[label] = Covariate(
            self,
            label,
            description,
            lambda trial: delta_stim(
                binfun(trial[stim_label]), binfun(trial.duration, True)
            ),
            **kwargs,
        )

    def add_covariate_spike(self, label, stim_label, description=None, basis=None):
        """
        Add a spike covariate with a default or provided basis.

        :param label: Covariate label.
        :type label: str
        :param stim_label: Label in trial dict for spike times.
        :type stim_label: str
        :param description: Description of the covariate.
        :type description: str or None
        :param basis: Basis object for the spike covariate, or None for default.
        :type basis: Basis or None
        """
        if basis is None:
            basis = make_nonlinear_raised_cos(
                10, self.experiment.time_unit_to_ms_ratio * self.experiment.binsize, (0., 100.), 1.
            )

        offset = basis.kwargs["nl_offset_in_ms"] / self.experiment.time_unit_to_ms_ratio
        assert offset > 0, (
            "offset must be greater than 0"
        )  # make sure causal. no instantaneous interaction
        binfun = self.experiment.binfun

        covar = Covariate(
            self,
            label,
            description,
            lambda trial: delta_stim(
                binfun(trial[stim_label]), binfun(trial.duration, True)
            ),
            basis,
            offset,
        )
        self.covariates[label] = covar

    def add_covariate_raw(self, label, description=None, **kwargs):
        """
        Add a raw (untransformed) covariate.

        :param label: Covariate label.
        :type label: str
        :param description: Description of the covariate.
        :type description: str or None
        :param kwargs: Additional arguments for Covariate.
        """
        self.covariates[label] = Covariate(
            self, label, description, raw_stim(label), **kwargs
        )

    def add_covariate_boxcar(
        self, label, on_label, off_label, description=None, **kwargs
    ):
        """
        Add a boxcar (rectangular) covariate.

        :param label: Covariate label.
        :type label: str
        :param on_label: Label in trial dict for boxcar start.
        :type on_label: str
        :param off_label: Label in trial dict for boxcar end.
        :type off_label: str
        :param description: Description of the covariate.
        :type description: str or None
        """
        binfun = self.experiment.binfun
        covar = Covariate(
            self,
            label,
            description,
            lambda trial: boxcar_stim(
                binfun(trial[on_label]),
                binfun(trial[off_label], True),  # NOTE: next bin if the event ocurred at the right bin edge
                binfun(trial.duration, True),
            ),
            **kwargs,
        )

        self.covariates[label] = covar

    def _filter_trials(self, trial_indices):
        """
        Internal helper to select trials by index.

        :param trial_indices: List of trial indices or None for all trials.
        :type trial_indices: list or None
        :returns: List of trial dicts.
        :rtype: list
        """
        expt = self.experiment
        if trial_indices is not None:
            trials = [expt.trials[idx] for idx in trial_indices]
        else:
            trials = expt.trials.values()
        return trials

    def get_response(self, label, trial_indices=None):
        """
        Get concatenated response vector for a label across selected trials.

        :param label: Trial dict key for the response.
        :type label: str
        :param trial_indices: List of trial indices or None for all trials.
        :type trial_indices: list or None
        :returns: Concatenated response array.
        :rtype: numpy.ndarray
        """
        trials = self._filter_trials(trial_indices)
        return np.concatenate([trial[label] for trial in trials])

    def get_binned_spike(
        self, label, trial_indices=None, concat=True
    ) -> NDArray:
        """
        Get binned spike counts for a label across selected trials.

        :param label: Trial dict key for spike times.
        :type label: str
        :param trial_indices: List of trial indices or None for all trials.
        :type trial_indices: list or None
        :param concat: Whether to concatenate results across trials.
        :type concat: bool
        :returns: Binned spike counts (concatenated array or list of arrays).
        :rtype: numpy.ndarray or list
        """
        trials = self._filter_trials(trial_indices)
        expt = self.experiment

        s = [
            _time2bin(trial[label], binwidth=expt.binsize, start=0, stop=trial.duration)
            for trial in trials
        ]
        s = np.concatenate(s)

        return s

    def compile_design_matrix(self, trial_indices=None) -> NDArray:
        """
        Compile the design matrix for selected trials.

        :param trial_indices: List of trial indices or None for all trials.
        :type trial_indices: list or None
        :returns: Design matrix of shape (total_bins, edim).
        :rtype: numpy.ndarray
        """
        binfun = self.experiment.binfun
        trials = self._filter_trials(trial_indices)

        X = []
        for trial in trials:
            n_bins = binfun(trial.duration, True)
            Xt = []
            for covar in self.covariates.values():
                if covar.condition is not None and not covar.condition(
                    trial
                ):  # skip trial
                    continue
                stim = covar.handler(trial)

                if stim.ndim == 1:
                    stim = np.expand_dims(stim, -1)  # column vector if 1D

                if covar.basis is None:
                    Xc = stim
                else:
                    Xc = conv_basis(
                        stim, covar.basis, ceil(covar.offset / self.experiment.binsize)
                    )
                Xt.append(Xc)
            Xt = np.column_stack(Xt)
            assert Xt.shape == (n_bins, self.edim)
            if not np.all(np.isfinite(Xt)):
                warnings.warn("Design matrix contains NaN or Inf")
            if self.bias:
                Xt = np.column_stack([np.ones(Xt.shape[0]), Xt])
            X.append(Xt)

        self._X = np.vstack(X)

        return self._X

    def combine_weights(self, w):
        """
        Split a weight vector into named fields for each covariate and combine with bases.

        :param w: Weight vector or matrix.
        :type w: numpy.ndarray
        :param axis: Axis along which to split weights.
        :type axis: int
        :returns: Named tuple of weight arrays for each covariate.
        :rtype: namedtuple
        """
        # TODO: constant column
        assert self.edim == len(w)

        if self.zstats:
            w = w * self.zstats['s'].squeeze() + self.zstats['m'].squeeze()
        
        sections = np.cumsum([covar.edim for covar in self.covariates.values()])[:-1]

        ws = np.array_split(
            w,
            sections,
        )
        
        binsize = self.experiment.binsize

        def covar_weight(covar: Covariate, w):
            basis = covar.basis
            
            if basis is None:
                tr = np.arange(len(w)) * binsize + covar.offset
                wout = w
            else:
                # combine weights and basis
                sdim = covar.edim // basis.edim  # raw variable dimension
                wout = np.zeros((basis.B.shape[0], sdim))
                for k in range(sdim):
                    wk = w[np.arange(basis.edim) + basis.edim * k]
                    wk2 = np.sum(wk * basis.B, -1)  # sum over bases
                    wout[:, k] = wk2
                if basis.tr.ndim == 2:
                    tr = basis.tr[:, 0]
                else:
                    tr = basis.tr

                tr = np.tile(tr[:, None] * binsize + covar.offset, (1, sdim))

            return {"label": covar.label, "tr": tr, "data": wout}

        w_dict = {covar.label: covar_weight(covar, wk) for covar, wk in zip(self.covariates.values(), ws)}
        
        return w_dict

    def get_design_matrix_col_indices(self, covar_labels: str | list[str]):
        """
        Get column indices in the design matrix for one or more covariates.

        :param covar_labels: Covariate label or list of labels.
        :type covar_labels: str or list of str
        :returns: Array of column indices corresponding to the covariates.
        :rtype: numpy.ndarray
        """
        if isinstance(covar_labels, str):
            covar_labels = [covar_labels]
        covars = self.covariates.values()
        csum = np.cumsum([covar.edim for covar in covars]).tolist()
        start = [0] + csum[:-1]
        end = csum

        indices = {
            covar.label: np.arange(start[k], end[k]) for k, covar in enumerate(covars)
        }

        col_indices = np.concatenate(
            [indices[covar_label] for covar_label in covar_labels]
        )
        return col_indices
    
    def zscore_columns(self, column_indices=None):
        X = self.X

        if column_indices is None:
            X, m, s = zscore(X)
        else:
            Z, mm, ss = zscore(X[:, column_indices])
            m = np.zeros((1, X.shape[1]))
            s = np.ones((1, X.shape[1]))
            m[:, column_indices] = mm
            s[:, column_indices] = ss
            X[:, column_indices] = Z
        
        self._X = X
        self.zstats['m'] = m
        self.zstats['s'] = s


@dataclass
class Covariate:
    """
    Covariate specification for a GLM design.

    :param design: The parent Design object.
    :type design: Design
    :param label: Covariate label.
    :type label: str
    :param description: Description of the covariate.
    :type description: str or None
    :param handler: Function to generate the stimulus array for each trial.
    :type handler: Callable
    :param basis: Basis object for the covariate, or None for raw.
    :type basis: Basis or None
    :param offset: Offset to apply to the covariate (in time units).
    :type offset: float
    :param condition: Optional function to filter trials for this covariate.
    :type condition: Callable or None
    :param sdim: Stimulus dimension (set automatically).
    :type sdim: int
    :param edim: Effective dimension (set automatically).
    :type edim: int
    """
    design: DesignMatrix
    label: str
    description: str | None
    handler: Callable
    basis: Basis | None = None
    offset: float = 0.0
    condition: Callable | None = None
    sdim: int = field(init=False)  # stimulus dimension
    edim: int = field(init=False)  # (covariate) effective dimension

    def __post_init__(self):
        if self.description is None:
            self.description = self.label

        trials = self.design.experiment.trials.values()
        assert len(trials) > 0

        template = self.handler(next(iter(trials)))
        if template.ndim == 1:
            sdim = 1
        else:
            sdim = np.size(template, 1)
        self.sdim = sdim

        if self.basis is None:
            edim = sdim
        else:
            edim = self.basis.edim * sdim
        self.edim = edim


def _time2bin(timing, binwidth, start, stop):
    """
    Bin event times into a histogram.

    :param timing: Array of event times.
    :type timing: array-like
    :param binwidth: Width of each bin.
    :type binwidth: float
    :param start: Start time of the first bin.
    :type start: float
    :param stop: End time of the last bin.
    :type stop: float
    :returns: Array of binned event counts.
    :rtype: numpy.ndarray
    """
    duration = stop - start
    n_bins = ceil(duration / binwidth)
    bin_edges = start + np.arange(n_bins + 1) * binwidth  # add the last bin edge
    s = np.histogram(timing, bins=bin_edges)[0]
    s = s.astype(float)
    return s


def raw_stim(label):
    """
    Return a handler function that extracts a raw stimulus array from a trial.

    :param label: Key in the trial dict for the stimulus.
    :type label: str
    :returns: Function that extracts the stimulus array from a trial.
    :rtype: Callable
    """
    return lambda t: t[label]


def constant_stim(label, binfun):
    return lambda t: boxcar_stim(0, binfun(t.duration, True), binfun(t.duration, True), t[label])
