import warnings
from math import ceil

import numpy as np

from .basis import conv_basis, delta_stim, raw_stim, boxcar_stim


class Design:
    def __init__(self, experiment):
        self.experiment = experiment
        self.covariates = {}

    @property
    def edim(self):
        return sum((covar.edim for covar in self.covariates))

    def add_covariate(self, label, description, handler, basis, offset, condition, **kwargs):
        self.covariates[label] = Covariate(label, description, handler, basis, offset, condition, **kwargs)

    def add_covariate_timing(self, label, description, *args, **kwargs):
        binfun = self.experiment.binfun
        self.covariates[label] = Covariate(label, description,
                                           lambda trial: delta_stim(binfun(trial[label]), binfun(trial.duration)),
                                           *args, **kwargs)

    def add_covariate_spike(self):
        raise NotImplementedError()

    def add_covariate_raw(self, label, description, *args, **kwargs):
        self.covariates[label] = Covariate(label, description,
                                           raw_stim(label),
                                           *args, **kwargs)

    def add_covariate_boxcar(self, label, description, on_label, off_label, *args, **kwargs):
        binfun = self.experiment.binfun
        self.covariates[label] = Covariate(label, description,
                                           lambda trial: boxcar_stim(binfun(trial[on_label]),
                                                                     binfun(trial[off_label]),
                                                                     binfun(trial.duration)),
                                           *args, **kwargs)

    def _filter_trials(self, trial_indices):
        expt = self.experiment
        if trial_indices is not None:
            trials = [expt.trials[idx] for idx in trial_indices]
        else:
            trials = expt.trials.values
        return trials

    def get_response(self, label, trial_indices):
        trials = self._filter_trials(trial_indices)
        rv = np.concatenate([trial[label] for trial in trials])
        return rv

    def compile_design_matrix(self, trial_indices):
        expt = self.experiment
        trials = self._filter_trials(trial_indices)
        # total_bins = sum(np.rint(np.ceil([trial.duration for trial in trials] / expt.binsize)))

        dm = []
        for trial in trials:
            nbin = ceil(trial.duration / expt.binsize)
            dmt = []
            for covar in self.covariates:
                if covar.condition is not None and not covar.condition(trial):  # skip trial
                    continue
                stim = covar.handler(trial, nbin)
                if covar.basis is None:
                    dmc = stim
                else:
                    dmc = conv_basis(stim, covar.basis, covar.offset)
                dmt.append(dmc)
            dmt = np.column_stack(dmt)
            assert dmt.shape == (nbin, self.edim)
            dm.append(dmt)
        dm = np.row_stack(dm)

        if np.any(np.isnan(dm)) or np.any(np.isinf(dm)):
            warnings.warn('Design matrix contains NaN or Inf')

        return dm


class Covariate:
    def __init__(self, design, label, description, handler, basis=None, offset=0, condition=None, **kwargs):
        self.design = design
        self.label = label
        self.description = description
        self.handler = handler
        self.basis = basis
        self.offset = offset
        self.condition = condition

        sdim = np.shape(handler(design.expt.trials.values[0]))[1]
        self.sdim = sdim

        if basis is None:
            edim = sdim
        else:
            edim = basis.edim * sdim
        self.edim = edim
