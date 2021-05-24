from collections import namedtuple

import numpy as np

from pyneuroglm.experiment import Experiment, Trial, Variable
from pyneuroglm.basis import make_smooth_temporal_basis, conv_basis


def test_experiment():
    expt = Experiment(time_unit='ms', binsize=10, eid=1, params=())
    assert expt.binfun(0.) == 1


def test_variable():
    v = Variable('label', 'description', 'type', 2)
    assert v.label == 'label' and v.description == 'description' and v.type == 'type' and v.ndim == 2


def test_trial():
    trial = Trial(1, 10)
    trial['a'] = np.zeros(100)
    assert trial['a'].shape == (100,)


def test_basis():
    expt = Experiment(time_unit='ms', binsize=10, eid=1, params=())
    B = make_smooth_temporal_basis('raised cosine', 100, 5, expt.binfun)
    print('\n', B.B)
    # plt.matshow(B.B)
    # plt.show()
    # plt.close()


def test_conv_basis():
    expt = Experiment(time_unit='ms', binsize=10, eid=1, params=())
    B = make_smooth_temporal_basis('raised cosine', 100, 5, expt.binfun)
    n = 100
    d = 2
    x = np.random.randn(n, d)
    X = conv_basis(x, B, offset=5)
    assert X.shape == (100, 10)
    X = conv_basis(x, B, offset=-5)
    assert X.shape == (100, 10)


def test_combine_weights():
    labels = ['a', 'b', 'c']
    values = [1, 2, 3]
    W = namedtuple('Weight', labels)
    W(*values)
