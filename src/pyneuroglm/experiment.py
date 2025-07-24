from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class VariableType(StrEnum):
    """
    Enumeration of variable types for experiment variables.

    - CONTINUOUS: Continuous-valued variable.
    - TIMING: Timing event variable.
    - VALUE: Value variable.
    - SPIKE: Spike train variable.
    """
    CONTINUOUS = "continuous"
    TIMING = "timing"
    VALUE = "value"
    SPIKE = "spike"


@dataclass
class Variable:
    """
    Specification for a variable in an experiment.

    :param label: Variable label.
    :type label: str
    :param description: Description of the variable.
    :type description: str
    :param type: Type of the variable (see VariableType).
    :type type: str
    :param ndim: Number of dimensions (default 1).
    :type ndim: int
    :param timing: Timing value for value variables (optional).
    :type timing: float or None
    """
    label: str
    description: str
    type: str
    ndim: int = 1
    timing: float | None = None


@dataclass
class Experiment:
    """
    Experiment specification and trial manager.

    :param time_unit: Time unit for the experiment ('ms', 's', etc.).
    :type time_unit: str
    :param binsize: Bin size for time discretization.
    :type binsize: float or int
    :param eid: Experiment identifier.
    :type eid: str
    :param meta: Optional metadata dictionary.
    :type meta: dict or None
    :param variables: Dictionary of variable label to Variable objects.
    :type variables: dict
    :param trials: Dictionary of trial id to Trial objects.
    :type trials: dict
    :param time_unit_to_ms_ratio: Conversion ratio from time_unit to milliseconds.
    :type time_unit_to_ms_ratio: float
    """
    time_unit: str
    binsize: float | int
    eid: str
    meta: dict | None = None
    variables: dict[Any, Variable] = field(default_factory=dict)
    trials: OrderedDict = field(default_factory=OrderedDict)
    time_unit_to_ms_ratio: float = 1.

    def __post_init__(self):
        """
        Set the time_unit_to_ms_ratio based on the time_unit.
        """
        match self.time_unit:
            case "ms":
                self.time_unit_to_ms_ratio = 1
            case "s":
                self.time_unit_to_ms_ratio = 1000
            case _:
                raise ValueError(f"Undefined time unit {self.time_unit}")

    def binfun(self, t: ArrayLike, right_edge: bool=False) -> Any:
        """
        Convert event time(s) to bin index or number of bins.

        :param t: Event time(s) as array-like.
        :type t: array-like
        :param right_edge: Use the right bin edge to determine, equal to number of bins indices.
        :type right_edge: bool
        :returns: Bin index/indices or number of bins.
        :rtype: int, numpy.ndarray, or scalar
        """
        t = np.asarray(t)
        assert np.all(t > 0)

        idx = t / self.binsize
        idx = np.maximum(np.ceil(idx), 1)
        idx = idx.astype(int)

        if not right_edge:
            idx -= 1

        if np.ndim(idx) == 0:
            idx = idx.item()
        
        return idx

    def register_continuous(self, label, description, ndim=1):
        """
        Register a continuous variable.

        :param label: Variable label.
        :type label: str
        :param description: Description of the variable.
        :type description: str
        :param ndim: Number of dimensions (default 1).
        :type ndim: int
        """
        self.variables[label] = Variable(label, description, 'continuous',
                                         ndim)

    def register_timing(self, label, description):
        """
        Register a timing variable.

        :param label: Variable label.
        :type label: str
        :param description: Description of the variable.
        :type description: str
        """
        self.variables[label] = Variable(label, description, 'value')

    def register_spike_train(self, label, description):
        """
        Register a spike train variable.

        :param label: Variable label.
        :type label: str
        :param description: Description of the variable.
        :type description: str
        """
        self.variables[label] = Variable(label, description, "spike")

    def register_value(self, label, description, timing=None):
        """
        Register a value variable.

        :param label: Variable label.
        :type label: str
        :param description: Description of the variable.
        :type description: str
        :param timing: Timing value for the variable (optional).
        :type timing: float or None
        """
        v = Variable(label, description, 'value')
        v.timing = timing
        self.variables[label] = v

    def add_trial(self, trial):
        """
        Add a trial to the experiment, checking variable registration and shape.

        :param trial: Trial object to add.
        :type trial: Trial
        :raises ValueError: If a variable in the trial is not registered.
        :raises AssertionError: If variable data does not match expected shape/type.
        """
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
    """
    Trial data container for an experiment.

    :param tid: Trial identifier.
    :type tid: Any
    :param duration: Duration of the trial.
    :type duration: float
    :param _variables: Dictionary of variable label to data (set automatically).
    :type _variables: dict
    """
    tid: Any
    duration: float
    _variables: dict = field(init=False, default_factory=dict)

    def __getitem__(self, key):
        """
        Get variable data by key.

        :param key: Variable label.
        :type key: str
        :returns: Variable data.
        """
        return self._variables[key]

    def __setitem__(self, key, value):
        """
        Set variable data by key.

        :param key: Variable label.
        :type key: str
        :param value: Data to set.
        """
        self._variables[key] = value

    @property
    def variables(self):
        """
        Get all variable (label, value) pairs for this trial.

        :returns: Items view of variable label to data.
        :rtype: dict_items
        """
        return self._variables.items()
