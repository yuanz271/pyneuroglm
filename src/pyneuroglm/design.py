from math import ceil
from typing import Any
import warnings
from collections.abc import Callable
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np

from .basis import conv_basis, delta_stim, boxcar_stim, make_nonlinear_raised_cos, Basis


__all__ = ["Design", "Covariate"]


@dataclass
class Design:
    experiment: Any
    covariates: dict = field(default_factory=dict)
    bias = False

    @property
    def edim(self):
        return sum((covar.edim for covar in self.covariates.values()))

    def add_constant(self, bias=True):
        self.bias = bias

    def add_covariate(
        self,
        label,
        description,
        handler,
        basis,
        offset=0,
        condition: Callable | None = None,
    ):
        self.covariates[label] = Covariate(
            self, label, description, handler, basis, offset, condition
        )

    def add_covariate_timing(self, label, stim_label=None, description=None, **kwargs):
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
        if description is None:
            description = label
        if basis is None:
            basis = make_nonlinear_raised_cos(
                10, self.experiment.binsize, (0.0, 0.1), self.experiment.binsize
            )

        offset = basis.kwargs["nl_offset"]
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
        self.covariates[label] = Covariate(
            self, label, description, raw_stim(label), **kwargs
        )

    def add_covariate_boxcar(
        self, label, on_label, off_label, description=None, value_label=None
    ):
        if description is None:
            description = label

        binfun = self.experiment.binfun
        if value_label is None:
            covar = Covariate(
                self,
                label,
                description,
                lambda trial: boxcar_stim(
                    binfun(trial[on_label]),
                    binfun(trial[off_label]),
                    binfun(trial.duration, True),
                ),
            )
        else:
            covar = Covariate(
                self,
                label,
                description,
                lambda trial: boxcar_stim(
                    binfun(trial[on_label]),
                    binfun(trial[off_label]),
                    binfun(trial.duration, True),
                    trial[value_label],
                ),
            )
        self.covariates[label] = covar

    def _filter_trials(self, trial_indices):
        expt = self.experiment
        if trial_indices is not None:
            trials = [expt.trials[idx] for idx in trial_indices]
        else:
            trials = expt.trials.values()
        return trials

    def get_response(self, label, trial_indices=None):
        trials = self._filter_trials(trial_indices)
        return np.concatenate([trial[label] for trial in trials])

    def get_binned_spike(
        self, label, trial_indices=None, concat=True
    ) -> np.ndarray | list:
        trials = self._filter_trials(trial_indices)
        expt = self.experiment

        s = [
            _time2bin(trial[label], binwidth=expt.binsize, start=0, stop=trial.duration)
            for trial in trials
        ]
        if concat:
            s = np.concatenate(s)

        return s

    def compile_design_matrix(self, trial_indices=None) -> np.ndarray:
        expt = self.experiment
        trials = self._filter_trials(trial_indices)

        dm = []
        for trial in trials:
            n_bins = expt.binfun(trial.duration, True)
            dmt = []
            for covar in self.covariates.values():
                if covar.condition is not None and not covar.condition(
                    trial
                ):  # skip trial
                    continue
                stim = covar.handler(trial)

                if stim.ndim == 1:
                    stim = np.expand_dims(stim, -1)  # column vector if 1D

                if covar.basis is None:
                    dmc = stim
                else:
                    dmc = conv_basis(
                        stim, covar.basis, ceil(covar.offset / expt.binsize)
                    )
                dmt.append(dmc)
            dmt = np.concatenate(dmt, axis=1)
            assert dmt.shape == (n_bins, self.edim)
            if np.any(np.isnan(dmt)) or np.any(np.isinf(dmt)):
                warnings.warn("Design matrix contains NaN or Inf")
            if self.bias:
                dmt = np.column_stack([np.ones(dmt.shape[0]), dmt])
            dm.append(dmt)

        dm = np.concatenate(dm, axis=0)

        return dm

    def combine_weights(self, w, axis=1):
        ws = np.split(
            w,
            np.cumsum([covar.edim for covar in self.covariates.values()])[:-1],
            axis=axis,
        )
        W = namedtuple("Weight", [covar.label for covar in self.covariates.values()])
        return W(*ws)

    def get_design_matrix_col_indices(self, covar_labels: str | list[str]):
        if isinstance(covar_labels, str):
            covar_labels = [covar_labels]
        covars = self.covariates.values()
        csum = np.cumsum([covar.edim for covar in covars]).tolist()
        # print(F"{csum=}")
        start = [0] + csum[:-1]
        end = csum

        # print(start, end)

        indices = {
            covar.label: np.arange(start[k], end[k]) for k, covar in enumerate(covars)
        }

        # print(indices)

        col_indices = np.concatenate(
            [indices[covar_label] for covar_label in covar_labels]
        )
        return col_indices


@dataclass
class Covariate:
    design: Design
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
    duration = stop - start
    n_bins = ceil(duration / binwidth)
    bin_edges = start + np.arange(n_bins + 1) * binwidth  # add the last bin edge
    s = np.histogram(timing, bins=bin_edges)[0]
    s = s.astype(float)
    return s


def raw_stim(label):
    return lambda t: t[label]
