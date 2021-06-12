from collections import namedtuple

import numpy as np

from pyneuroglm.experiment import Experiment, Trial, Variable
from pyneuroglm.basis import make_smooth_temporal_basis, conv_basis, make_nonlinear_raised_cosine, \
    nonlinear_raised_cosine


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


def test_make_smooth_temporal_basis():
    expt = Experiment(time_unit='ms', binsize=10, eid=1, params=())
    basis = make_smooth_temporal_basis('raised cosine', 100, 5, expt.binfun)
    basis_boot = basis.func(*basis.args)
    assert np.all(basis.B == basis_boot.B)
    # print('\n', B.B)
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
    w = W(*values)
    assert w == (1, 2, 3)


def test_make_nonlinear_raised_cosine():
    basis_matlab = np.load('basis.npy', allow_pickle=True)
    B = basis_matlab['B'][()]
    param = basis_matlab['param'][()]  # dtype=[('nBases', 'O'), ('binSize', 'O'), ('endPoints', 'O'), ('nlOffset', 'O')])
    nBases, binSize, endPoints, nlOffset = param.tolist()

    basis = make_nonlinear_raised_cosine(nBases, binSize, endPoints, nlOffset)
    assert np.allclose(basis.B, B)

    basis_boot = basis.func(*basis.args)
    assert np.all(basis.B == basis_boot.B)
