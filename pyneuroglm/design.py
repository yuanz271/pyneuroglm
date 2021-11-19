from math import ceil
import warnings
from collections import namedtuple

import numpy as np

from .basis import conv_basis, delta_stim, boxcar_stim, make_nonlinear_raised_cosine

__all__ = ['Design', 'Covariate']


class Design:
    covariates = {}
    bias = False

    def __init__(self, experiment):
        self.experiment = experiment

    @property
    def edim(self):
        return sum((covar.edim for covar in self.covariates.values()))

    def add_constant(self, bias=True):
        self.bias = bias

    def add_covariate(self, label, description, handler, basis, offset,
                      condition, **kwargs):
        self.covariates[label] = Covariate(self, label, description, handler,
                                           basis, offset, condition, **kwargs)

    def add_covariate_timing(self, label, description, var_label, value_label,
                             *args, **kwargs):
        binfun = self.experiment.binfun
        if value_label is None:
            self.covariates[label] = Covariate(
                self, label, description, lambda trial: delta_stim(
                    binfun(trial[var_label]), binfun(trial.duration)), *args,
                **kwargs)
        else:
            self.covariates[label] = Covariate(
                self, label, description,
                lambda trial: trial[value_label] * delta_stim(
                    binfun(trial[var_label]), binfun(trial.duration)), *args,
                **kwargs)

    def add_covariate_spike(self, label, description, var_label, basis,
                            **kwargs):
        offset = 1  # make sure causal. no instantaneous interaction
        binfun = self.experiment.binfun
        if basis is None:
            basis = make_nonlinear_raised_cosine(10, self.experiment.binsize,
                                                 (0., 100.), 2)
        covar = Covariate(
            self, label, description, lambda trial: delta_stim(
                binfun(trial[var_label]), binfun(trial.duration)), basis,
            offset, **kwargs)
        self.covariates[label] = covar

    def add_covariate_raw(self, label, description, *args, **kwargs):
        self.covariates[label] = Covariate(self, label, description,
                                           lambda trial: trial[label], *args,
                                           **kwargs)

    def add_covariate_boxcar(self, label, description, on_label, off_label,
                             value_label, *args, **kwargs):
        binfun = self.experiment.binfun
        if value_label is None:
            covar = Covariate(
                self, label, description, lambda trial: boxcar_stim(
                    binfun(trial[on_label]), binfun(trial[off_label]),
                    binfun(trial.duration)), *args, **kwargs)
        else:
            covar = Covariate(
                self, label, description,
                lambda trial: trial[value_label] * boxcar_stim(
                    binfun(trial[on_label]), binfun(trial[off_label]),
                    binfun(trial.duration)), *args, **kwargs)
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
        # print(sum([trial[label].shape[0] for trial in trials]),
        #       sum([self.experiment.binfun(trial.duration) for trial in trials]))
        return np.concatenate([trial[label] for trial in trials])

    def get_binned_spike(self, label, trial_indices=None, concat=True):
        trials = self._filter_trials(trial_indices)
        expt = self.experiment

        s = [
            _time2bin(trial[label],
                      binwidth=expt.binsize,
                      start=0,
                      stop=trial.duration) for trial in trials
        ]
        if concat:
            s = np.concatenate(s)

        return s

    def compile_design_matrix(self, trial_indices=None, concat=True):
        expt = self.experiment
        trials = self._filter_trials(trial_indices)
        # total_bins = sum([expt.binfun(trial.duration) for trial in trials])
        # print(total_bins)

        dm = []
        for trial in trials:
            nbin = expt.binfun(trial.duration)
            dmt = []
            for covar in self.covariates.values():
                if covar.condition is not None and not covar.condition(
                        trial):  # skip trial
                    continue
                stim = covar.handler(trial)
                if covar.basis is None:
                    dmc = stim
                else:
                    dmc = conv_basis(stim, covar.basis, covar.offset)
                    # print(dmc.shape)
                dmt.append(dmc)
            dmt = np.concatenate(dmt, axis=1)
            assert dmt.shape == (nbin, self.edim)
            if np.any(np.isnan(dmt)) or np.any(np.isinf(dmt)):
                warnings.warn('Design matrix contains NaN or Inf')
            if self.bias:
                dmt = np.column_stack([np.ones(dmt.shape[0]), dmt])
            dm.append(dmt)
        if concat:
            dm = np.concatenate(dm, axis=0)

        return dm

    def combine_weights(self, w, axis=1):
        ws = np.split(
            w,
            np.cumsum([covar.edim for covar in self.covariates.values()])[:-1],
            axis=axis)
        W = namedtuple('Weight',
                       [covar.label for covar in self.covariates.values()])
        return W(*ws)


class Covariate:
    def __init__(self,
                 design,
                 label,
                 description,
                 handler,
                 basis=None,
                 offset=0,
                 condition=None,
                 **kwargs):
        self.design = design
        self.label = label
        self.description = description
        self.handler = handler  # function of trial
        self.basis = basis
        self.offset = offset
        self.condition = condition

        sdim = np.shape(handler(next(iter(
            design.experiment.trials.values()))))[1]
        self.sdim = sdim

        if basis is None:
            edim = sdim
        else:
            edim = basis.edim * sdim
        self.edim = edim


def _time2bin(timing, binwidth, start, stop):
    duration = stop - start
    nbin = ceil(duration / binwidth)
    bins = start + np.arange(nbin + 1) * binwidth  # add the last bin edge
    s = np.histogram(timing, bins=bins)[0]
    s = s.astype(np.float)
    return s
