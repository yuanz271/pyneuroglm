"""
Design matrix construction for neuroGLM-style experiments.

Defines `DesignMatrix` and `Covariate` to assemble covariates, apply temporal
bases, and compile per-trial inputs into a single matrix suitable for GLM
fitting.
"""

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

    Parameters
    ----------
    experiment : Experiment
        The Experiment object containing trial data and binning functions.
    covariates : dict, optional
        Dictionary of covariate label to Covariate objects.
    bias : bool, optional
        Whether to include a bias (constant) column in the design matrix.
    zstats : dict, optional
        Dictionary of z-scoring statistics.
    """

    experiment: Experiment
    bias: bool = False
    covariates: dict = field(init=False, default_factory=dict)
    zstats: dict = field(init=False, default_factory=dict)
    _X: NDArray | None = field(init=False, default=None)

    @property
    def X(self) -> NDArray:
        """
        The compiled design matrix.

        Returns
        -------
        numpy.ndarray
            The compiled design matrix.

        Raises
        ------
        RuntimeError
            If the design matrix has not been compiled.
        """
        if self._X is None:
            raise RuntimeError("Design matrix has not been compiled")
        else:
            return self._X

    @property
    def edim(self):
        """
        Total effective dimension (number of columns) of the design matrix.

        Returns
        -------
        int
            Sum of effective dimensions of all covariates.
        """
        return sum((covar.edim for covar in self.covariates.values()))

    def prepend_constant(self, bias=True):
        """
        Add a constant (bias) column to the design matrix.

        Parameters
        ----------
        bias : bool, optional
            Whether to include the bias column. Default is True.

        Raises
        ------
        NotImplementedError
            This method is not implemented.
        """
        raise NotImplementedError
        # self.bias = bias

    def add_covariate(
        self,
        label,
        description,
        handler,
        basis=None,
        offset=0.0,
        condition: Callable | None = None,
    ):
        """
        Add a covariate to the design.

        Parameters
        ----------
        label : str
            Covariate label.
        description : str
            Description of the covariate.
        handler : Callable
            Function to generate the stimulus array for each trial.
        basis : Basis or None, optional
            Basis object for the covariate, or None for raw.
        offset : float, optional
            Offset to apply to the covariate (in time units).
        condition : Callable or None, optional
            Optional function to filter trials for this covariate.
        """
        self.covariates[label] = Covariate(
            self, label, description, handler, basis, offset, condition
        )

    def add_covariate_constant(self, label, stim_label=None, description=None, **kwargs):
        """
        Add a constant covariate.

        Parameters
        ----------
        label : str
            Covariate label.
        stim_label : str or None, optional
            Label in trial dict for the constant value (defaults to label).
        description : str or None, optional
            Description of the covariate.
        **kwargs
            Additional arguments for Covariate.
        """
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

        Parameters
        ----------
        label : str
            Covariate label.
        stim_label : str or None, optional
            Label in trial dict for event times (defaults to label).
        description : str or None, optional
            Description of the covariate.
        **kwargs
            Additional arguments for Covariate.
        """
        binfun = self.experiment.binfun
        if stim_label is None:
            stim_label = label

        self.covariates[label] = Covariate(
            self,
            label,
            description,
            lambda trial: delta_stim(binfun(trial[stim_label]), binfun(trial.duration, True)),
            **kwargs,
        )

    def add_covariate_spike(self, label, stim_label, description=None, basis=None):
        """
        Add a spike covariate with a default or provided basis.

        Parameters
        ----------
        label : str
            Covariate label.
        stim_label : str
            Label in trial dict for spike times.
        description : str or None, optional
            Description of the covariate.
        basis : Basis or None, optional
            Basis object for the spike covariate, or None for default.
        """
        if basis is None:
            basis = make_nonlinear_raised_cos(
                10,
                self.experiment.time_unit_to_ms_ratio * self.experiment.binsize,
                (0.0, 100.0),
                1.0,
            )

        offset = basis.kwargs["nl_offset_in_ms"] / self.experiment.time_unit_to_ms_ratio
        assert (
            offset > 0
        ), "offset must be greater than 0"  # make sure causal. no instantaneous interaction
        binfun = self.experiment.binfun

        covar = Covariate(
            self,
            label,
            description,
            lambda trial: delta_stim(binfun(trial[stim_label]), binfun(trial.duration, True)),
            basis,
            offset,
        )
        self.covariates[label] = covar

    def add_covariate_raw(self, label, description=None, **kwargs):
        """
        Add a raw (untransformed) covariate.

        Parameters
        ----------
        label : str
            Covariate label.
        description : str or None, optional
            Description of the covariate.
        **kwargs
            Additional arguments for Covariate.
        """
        self.covariates[label] = Covariate(self, label, description, raw_stim(label), **kwargs)

    def add_covariate_boxcar(self, label, on_label, off_label, description=None, **kwargs):
        """
        Add a boxcar (rectangular) covariate.

        Parameters
        ----------
        label : str
            Covariate label.
        on_label : str
            Label in trial dict for boxcar start.
        off_label : str
            Label in trial dict for boxcar end.
        description : str or None, optional
            Description of the covariate.
        **kwargs
            Additional arguments for Covariate.
        """
        binfun = self.experiment.binfun
        covar = Covariate(
            self,
            label,
            description,
            lambda trial: boxcar_stim(
                binfun(trial[on_label]),
                binfun(
                    trial[off_label], True
                ),  # NOTE: next bin if the event ocurred at the right bin edge
                binfun(trial.duration, True),
            ),
            **kwargs,
        )

        self.covariates[label] = covar

    def _filter_trials(self, trial_indices):
        """
        Select trials by index.

        Parameters
        ----------
        trial_indices : list or None
            List of trial indices or None for all trials.

        Returns
        -------
        list
            List of trial dicts.
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

        Parameters
        ----------
        label : str
            Trial dict key for the response.
        trial_indices : list or None, optional
            List of trial indices or None for all trials.

        Returns
        -------
        numpy.ndarray
            Concatenated response array.
        """
        trials = self._filter_trials(trial_indices)
        return np.concatenate([trial[label] for trial in trials])

    def get_binned_spike(self, label, trial_indices=None, concat=True) -> NDArray:
        """
        Get binned spike counts for a label across selected trials.

        Parameters
        ----------
        label : str
            Trial dict key for spike times.
        trial_indices : list or None, optional
            List of trial indices or None for all trials.
        concat : bool, optional
            Ignored; results are always concatenated across trials.

        Returns
        -------
        numpy.ndarray
            1D array of binned spike counts concatenated across trials.

        Notes
        -----
        The `concat` parameter is reserved for future use; the current
        implementation always concatenates per-trial results.
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

        Parameters
        ----------
        trial_indices : list or None, optional
            List of trial indices or None for all trials.

        Returns
        -------
        numpy.ndarray
            Design matrix of shape (total_bins, edim).
        """
        binfun = self.experiment.binfun
        trials = self._filter_trials(trial_indices)

        X = []
        for trial in trials:
            n_bins = binfun(trial.duration, True)
            Xt = []
            for covar in self.covariates.values():
                if covar.condition is not None and not covar.condition(trial):  # skip trial
                    continue
                stim = covar.handler(trial)

                if stim.ndim == 1:
                    stim = np.expand_dims(stim, -1)  # column vector if 1D

                if covar.basis is None:
                    Xc = stim
                else:
                    Xc = conv_basis(stim, covar.basis, ceil(covar.offset / self.experiment.binsize))
                Xt.append(Xc)
            Xt = np.column_stack(Xt)
            assert Xt.shape == (n_bins, self.edim)
            if not np.all(np.isfinite(Xt)):
                warnings.warn("Design matrix contains NaN or Inf")
            # if self.bias:
            #     Xt = np.column_stack([np.ones(Xt.shape[0]), Xt])
            X.append(Xt)

        self._X = np.vstack(X)

        return self._X

    def combine_weights(self, w):
        """
        Split a weight vector into named fields for each covariate and combine with bases.

        Parameters
        ----------
        w : numpy.ndarray
            1D weight vector of shape (edim,).

        Returns
        -------
        dict
            Dictionary mapping covariate labels to weight arrays and metadata.
        """
        # TODO: constant column
        assert self.edim == len(w)

        if self.zstats:
            w = w * self.zstats["s"].squeeze() + self.zstats["m"].squeeze()

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

        w_dict = {
            covar.label: covar_weight(covar, wk) for covar, wk in zip(self.covariates.values(), ws)
        }

        return w_dict

    def get_design_matrix_col_indices(self, covar_labels: str | list[str]):
        """
        Get column indices in the design matrix for one or more covariates.

        Parameters
        ----------
        covar_labels : str or list of str
            Covariate label or list of labels.

        Returns
        -------
        numpy.ndarray
            Array of column indices corresponding to the covariates.
        """
        if isinstance(covar_labels, str):
            covar_labels = [covar_labels]
        covars = self.covariates.values()
        csum = np.cumsum([covar.edim for covar in covars]).tolist()
        start = [0] + csum[:-1]
        end = csum

        indices = {covar.label: np.arange(start[k], end[k]) for k, covar in enumerate(covars)}

        col_indices = np.concatenate([indices[covar_label] for covar_label in covar_labels])
        return col_indices

    def zscore_columns(self, column_indices=None):
        """
        Z-score specified columns of the design matrix.

        Parameters
        ----------
        column_indices : array-like or None, optional
            Indices of columns to z-score. If None, all columns are z-scored.
        """
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
        self.zstats["m"] = m
        self.zstats["s"] = s


@dataclass
class Covariate:
    """
    Covariate specification for a GLM design.

    Parameters
    ----------
    design : DesignMatrix
        The parent DesignMatrix object.
    label : str
        Covariate label.
    description : str or None
        Description of the covariate.
    handler : Callable
        Function to generate the stimulus array for each trial.
    basis : Basis or None, optional
        Basis object for the covariate, or None for raw.
    offset : float, optional
        Offset to apply to the covariate (in time units).
    condition : Callable or None, optional
        Optional function to filter trials for this covariate.
    sdim : int
        Stimulus dimension (set automatically).
    edim : int
        Effective dimension (set automatically).
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
        """Initialize derived dimensions from a template trial.

        Infers stimulus dimension (`sdim`) from the handler's output on a
        template trial, and sets effective dimension (`edim`) based on the
        presence of a temporal basis.
        """
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

    Parameters
    ----------
    timing : array-like
        Array of event times.
    binwidth : float
        Width of each bin.
    start : float
        Start time of the first bin.
    stop : float
        End time of the last bin.

    Returns
    -------
    numpy.ndarray
        Array of binned event counts.
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

    Parameters
    ----------
    label : str
        Key in the trial dict for the stimulus.

    Returns
    -------
    Callable
        Function that extracts the stimulus array from a trial.
    """
    return lambda t: t[label]


def constant_stim(label, binfun):
    """
    Return a handler function that creates a constant stimulus array for a trial.

    Parameters
    ----------
    label : str
        Key in the trial dict for the stimulus.
    binfun : Callable
        Function to compute the number of bins.

    Returns
    -------
    Callable
        Function that creates a constant stimulus array for a trial.
    """
    return lambda t: boxcar_stim(0, binfun(t.duration, True), binfun(t.duration, True), t[label])
