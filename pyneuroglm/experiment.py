from math import ceil


class Variable:
    def __init__(self, label, description, vtype, ndim=1):
        self.label = label
        self.description = description
        self.type = vtype
        self.ndim = ndim


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
        assert t >= 0.
        return max(1, ceil(t / self.binsize))

    def register_continuous(self, label, description, ndim=1):
        self.variables[label] = Variable(label, description, 'continuous', ndim)

    def register_timing(self, label, description):
        self.variables[label] = Variable(label, description, 'value')

    def register_spike(self):
        pass

    def register_value(self, label, description, timing):
        v = Variable(label, description, 'value')
        v.timing = timing
        self.variables[label] = v

    def add_trial(self, trial, tid):
        for label, v in trial.variables:
            if label not in self.variables:
                raise ValueError('Unregistered variable')
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
        self.trials[tid] = trial

    def get_binned_spike(self):
        pass

    def get_response_variable(self):
        pass


class Trial:
    def __init__(self, tid, duration):
        self.tid = tid
        self.duration = duration
        self.variables = {}
