from collections import namedtuple
from pathlib import Path

import numpy as np

from pyneuroglm.basis import make_smooth_temporal_basis, conv_basis, make_nonlinear_raised_cos
from pyneuroglm.experiment import Experiment, Trial, Variable


def test_experiment():
    expt = Experiment(time_unit='ms', binsize=10, eid=1, params=())
    assert expt.binfun(0.) == 1


def test_variable():
    v = Variable('label', 'description', 'type', 2)
    assert v.label == 'label' and v.description == 'description' and v.type == 'type' and v.ndim == 2


def test_trial():
    trial = Trial(1, 10)
    trial['a'] = np.zeros(100)
    assert trial['a'].shape == (100, )


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


def test_make_nonlinear_raised_cos():
    basis_matlab = np.load(Path(__file__).parent / 'basis.npy', allow_pickle=True)
    B = basis_matlab['B'][()]
    param = basis_matlab['param'][(
    )]  # dtype=[('nBases', 'O'), ('binSize', 'O'), ('endPoints', 'O'), ('nlOffset', 'O')])
    n_bases, binsize, end_points, nl_offset = param.tolist()

    basis = make_nonlinear_raised_cos(n_bases, binsize, end_points, nl_offset)
    assert np.allclose(basis.B[:243, :], B)

    basis_recons = basis.func(*basis.args)
    assert np.all(basis.B == basis_recons.B)
