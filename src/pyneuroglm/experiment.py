from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class VariableType(StrEnum):
    """
    Enumeration of variable types for experiment variables.

    Attributes
    ----------
    CONTINUOUS : str
        Continuous-valued variable.
    TIMING : str
        Timing event variable.
    VALUE : str
        Value variable.
    SPIKE : str
        Spike train variable.
    """
    CONTINUOUS = "continuous"
    TIMING = "timing"
    VALUE = "value"
    SPIKE = "spike"


@dataclass
class Variable:
    """
    Specification for a variable in an experiment.

    Parameters
    ----------
    label : str
        Variable label.
    description : str
        Description of the variable.
    type : str
        Type of the variable (see VariableType).
    ndim : int, optional
        Number of dimensions (default 1).
    timing : float or None, optional
        Timing value for value variables (optional).
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

    Parameters
    ----------
    time_unit : str
        Time unit for the experiment ('ms', 's', etc.).
    binsize : float or int
        Bin size for time discretization.
    eid : str
        Experiment identifier.
    meta : dict or None, optional
        Optional metadata dictionary.
    variables : dict, optional
        Dictionary of variable label to Variable objects.
    trials : OrderedDict, optional
        Dictionary of trial id to Trial objects.
    time_unit_to_ms_ratio : float, optional
        Conversion ratio from time_unit to milliseconds.
    """
    time_unit: str
    binsize: float | int
    eid: Any
    meta: dict = field(default_factory=dict)
    variables: dict[Any, Variable] = field(default_factory=dict)
    trials: OrderedDict = field(default_factory=OrderedDict)
    time_unit_to_ms_ratio: float = field(init=False, default=1.)

    def __post_init__(self):
        """
        Set the time_unit_to_ms_ratio based on the time_unit.

        Raises
        ------
        ValueError
            If the time unit is not recognized.
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

        Parameters
        ----------
        t : array-like
            Event time(s).
        right_edge : bool, optional
            If True, return number of bins. If False, return bin indices. Default is False.

        Returns
        -------
        int, numpy.ndarray, or scalar
            Bin index/indices or number of bins.

        Raises
        ------
        AssertionError
            If any event time is negative.
        """
        t = np.asarray(t)
        assert np.all(t >= 0)

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

        Parameters
        ----------
        label : str
            Variable label.
        description : str
            Description of the variable.
        ndim : int, optional
            Number of dimensions (default 1).
        """
        self.variables[label] = Variable(label, description, 'continuous',
                                         ndim)

    def register_timing(self, label, description):
        """
        Register a timing variable.

        Parameters
        ----------
        label : str
            Variable label.
        description : str
            Description of the variable.
        """
        self.variables[label] = Variable(label, description, 'timing')

    def register_spike_train(self, label, description):
        """
        Register a spike train variable.

        Parameters
        ----------
        label : str
            Variable label.
        description : str
            Description of the variable.
        """
        self.variables[label] = Variable(label, description, "spike")

    def register_value(self, label, description, ndim=1, timing=None):
        """
        Register a value variable.

        Parameters
        ----------
        label : str
            Variable label.
        description : str
            Description of the variable.
        ndim : int, optional
            Number of dimensions (default 1).
        timing : float or None, optional
            Timing value for the variable.
        """
        v = Variable(label, description, 'value', ndim)
        v.timing = timing
        self.variables[label] = v

    def add_trial(self, trial):
        """
        Add a trial to the experiment, checking variable registration and shape.

        Parameters
        ----------
        trial : Trial
            Trial object to add.

        Raises
        ------
        ValueError
            If a variable in the trial is not registered.
        AssertionError
            If variable data does not match expected shape/type.
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

    Parameters
    ----------
    tid : Any
        Trial identifier.
    duration : float
        Duration of the trial.
    _variables : dict, optional
        Dictionary of variable label to data (set automatically).
    """
    tid: Any
    duration: float
    _variables: dict = field(init=False, default_factory=dict)

    def __getitem__(self, key):
        """
        Get variable data by key.

        Parameters
        ----------
        key : str
            Variable label.

        Returns
        -------
        Any
            Variable data.
        """
        return self._variables[key]

    def __setitem__(self, key, value):
        """
        Set variable data by key.

        Parameters
        ----------
        key : str
            Variable label.
        value : Any
            Data to set.
        """
        self._variables[key] = value

    @property
    def variables(self):
        """
        Get all variable (label, value) pairs for this trial.

        Returns
        -------
        dict_items
            Items view of variable label to data.
        """
        return self._variables.items()
