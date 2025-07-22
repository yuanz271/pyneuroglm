from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np


class VariableType(StrEnum):
    CONTINUOUS = "continuous"
    TIMING = "timing"
    VALUE = "value"
    SPIKE = "spike"


@dataclass
class Variable:
    label: str
    description: str
    type: str
    ndim: int = 1
    timing: float | None = None


@dataclass
class Experiment:
    time_unit: str
    binsize: float | int
    eid: str
    meta: dict | None = None
    variables: dict[Any, Variable] = field(default_factory=dict)
    trials: dict = field(default_factory=dict)

    def binfun(self, t, return_n_bins=False):
        """Event time to bin index"""
        assert np.all(t >= 0.)
        idx = t / self.binsize
        if return_n_bins:
            idx = np.ceil(idx)
        return np.int_(idx)
    
    def time_unit_to_ms_ratio(self):
        """Convert time to millisecond"""
        match self.time_unit:
            case "ms":
                return 1
            case "s":
                return 1000
            case _:
                raise ValueError(f"Undefined time unit {self.time_unit}")

    def register_continuous(self, label, description, ndim=1):
        self.variables[label] = Variable(label, description, 'continuous',
                                         ndim)

    def register_timing(self, label, description):
        self.variables[label] = Variable(label, description, 'value')

    def register_spike_train(self, label, description):
        self.variables[label] = Variable(label, description, "spike")

    def register_value(self, label, description, timing=None):
        v = Variable(label, description, 'value')
        v.timing = timing
        self.variables[label] = v

    def add_trial(self, trial):
        for label, v in trial.variables:
            if label not in self.variables:
                raise ValueError(f'Unregistered variable: {label}')
                        
            match self.variables[label].type:
                case VariableType.CONTINUOUS:
                    assert self.binfun(trial.duration, True) == v.shape[0]
                case VariableType.TIMING:
                    assert v.ndim == 1
                    assert min(v) >= 0. and max(v) < trial.duration
                
        self.trials[trial.tid] = trial


@dataclass
class Trial:
    tid: Any
    duration: float
    _variables: dict = field(init=False, default_factory=dict)

    def __getitem__(self, key):
        return self._variables[key]

    def __setitem__(self, key, value):
        self._variables[key] = value

    @property
    def variables(self):
        return self._variables.items()
