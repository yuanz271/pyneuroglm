from matplotlib import pyplot as plt


from pyneuroglm.experiment import Experiment, Trial
from pyneuroglm.basis import make_smooth_temporal_basis


def test_experiment():
    expt = Experiment(time_unit='ms', binsize=10, eid=1, params=())

    assert expt.binfun(0.) == 1


def test_trial():
    trial = Trial(1, 10)


def test_basis():
    expt = Experiment(time_unit='ms', binsize=10, eid=1, params=())
    B = make_smooth_temporal_basis('raised cosine', 100, 5, expt.binfun)
    print('\n', B.B)
    plt.matshow(B.B)
    plt.show()
    plt.close()
