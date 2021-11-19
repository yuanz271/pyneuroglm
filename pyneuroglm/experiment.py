from dataclasses import dataclass

import numpy as np


@dataclass
class Variable:
    label: str
    description: str
    type: str
    ndim: int = 1


class Experiment:
    def __init__(self, time_unit, binsize, eid, params=None):
        assert isinstance(time_unit, str)
        assert binsize > 0

        self.time_unit = time_unit
        self.binsize = binsize
        self.eid = eid
        self.params = params

        self.variables = {}
        self.trials = {}

    def binfun(self, t):
        assert np.all(t >= 0.)
        return np.maximum(1, np.ceil(t / self.binsize)).astype(int)  # number of bins, minus 1 for bin index

    def register_continuous(self, label, description, ndim=1):
        self.variables[label] = Variable(label, description, 'continuous',
                                         ndim)

    def register_timing(self, label, description):
        self.variables[label] = Variable(label, description, 'value')

    def register_spike(self):
        raise NotImplementedError()

    def register_value(self, label, description, timing):
        v = Variable(label, description, 'value')
        v.timing = timing
        self.variables[label] = v

    def add_trial(self, trial):
        for label, v in trial.variables:
            if label not in self.variables:
                raise ValueError(f'Unregistered variable: {label}')
            elif self.variables[label].type == 'continuous':
                assert self.binfun(trial.duration) == v.shape[0]
            elif self.variables[label].type == 'timing':
                assert v.ndim == 1
                assert min(v) >= 0. and max(v) < trial.duration
            elif self.variables[label].type == 'value':
                pass
            elif self.variables[label].type == 'spike':
                pass
            else:
                raise ValueError('Unknown type')
        self.trials[trial.tid] = trial


class Trial:
    def __init__(self, tid, duration):
        self.tid = tid
        self.duration = duration
        self._variables = {}

    def __getitem__(self, key):
        return self._variables[key]

    def __setitem__(self, key, value):
        self._variables[key] = value

    @property
    def variables(self):
        return self._variables.items()
