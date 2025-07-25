from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pymatreader import read_mat
from sklearn.linear_model import PoissonRegressor
from pyneuroglm import basis
from pyneuroglm.experiment import Experiment, Trial
from pyneuroglm.design import DesignMatrix


def print_sparse(x):
    from scipy.sparse import coo_matrix
    if x.ndim == 1:
        x = np.expand_dims(x, -1)
    print(coo_matrix(x))


def main():
    matfile = read_mat(Path(__file__).parent / "exampleData.mat")
    print(matfile.keys())

    nTrials = matfile['nTrials']
    param = matfile['param']
    trials = matfile['trial']

    print(nTrials)
    print(type(param))
    print(param.keys())
    print(type(trials))
    print(trials.keys())

    # sptrain = trials['sptrain']  # list, nTtrials
    
    unit_of_time = "ms"
    binsize = 1  # 1ms
    time_scale = 1
    
    # %% Set up experiment
    expt = Experiment(unit_of_time, binsize, "", param)
    expt.register_continuous('LFP', 'Local Field Potential', 1)
    expt.register_continuous('eyepos', "Eye Position", 2)
    expt.register_timing('dotson', "Motion Dots Onset")  # timing variables are for events
    expt.register_timing('dotsoff', "Motion Dots Offset")
    expt.register_timing('saccade', "Saccade Time")
    expt.register_spike_train('sptrain', "Our Neuron")
    expt.register_spike_train('sptrain2', "Neighbor Neuron")
    expt.register_value('coh', "Coherence")  # trialwise info
    expt.register_value('choice', 'Direction of Choice')
    
    for k in range(nTrials):
        trial = Trial(k, trials['duration'][k] * time_scale)
        trial['LFP'] = trials['LFP'][k]
        trial['eyepos'] = trials['eyepos'][k]
        trial['dotson'] = trials['dotson'][k] * time_scale
        trial['dotsoff'] = trials['dotsoff'][k] * time_scale
        trial['saccade'] = trials['saccade'][k] * time_scale
        trial['coh'] = trials['coh'][k]
        trial['choice'] = trials['choice'][k]
        trial['sptrain'] = trials['sptrain'][k] * time_scale
        trial['sptrain2'] = trials['sptrain2'][k] * time_scale

        expt.add_trial(trial)
    
    # %% Set up design matrix
    dm = DesignMatrix(expt)
    binfun = expt.binfun
    bs = basis.make_smooth_temporal_basis('boxcar', 100 * time_scale, 10, binfun)
    bs.B = 0.1 * bs.B
    dm.add_covariate_raw('LFP', basis=bs)

    # %% Spike trains
    dm.add_covariate_spike('hist', 'sptrain', "History filter")
    dm.add_covariate_spike('coupling', 'sptrain2', "Coupling from neuron 2")
 
    # %% Dots
    dm.add_covariate_boxcar('dots', 'dotson', 'dotsoff', "Motion dots stim")

    # %% Saccade timing. Acausal (spike causes behavior)
    bs = basis.make_smooth_temporal_basis('boxcar', 300 * time_scale, 8, binfun)
    offset = -200 * time_scale
    dm.add_covariate_timing('saccade', basis=bs, offset=offset)

    # %% Coherence
    bs = basis.make_smooth_temporal_basis('raised cosine', 200 * time_scale, 10, binfun)
    def stim_handle(t):
        return t['coh'] * basis.boxcar_stim(binfun(t['dotson']), binfun(t['dotsoff'], True), binfun(t.duration, True))
    dm.add_covariate('cohKer', 'coh-dep dots stimulus', stim_handle, bs)
    
    # %% Eye position
    bs = basis.make_smooth_temporal_basis('raised cosine', 40 * time_scale, 4, binfun)
    dm.add_covariate_raw('eyepos', basis=bs)
    
    # %% Compile
    trial_indices = np.arange(10)
    dm.compile_design_matrix(trial_indices)

    end_trial_indices = np.cumsum([binfun(trial.duration, True) for trial in expt.trials.values()])
    X = dm.X[:end_trial_indices[2]]
    mv = np.max(np.abs(X), 0, keepdims=True)
    mv[np.isnan(mv)] = 1
    X = X / (mv + 1e-20)
    print(f"{X.shape=}")
    
    # %% Visualize design matrix
    fig, ax = plt.subplots()
    ax.imshow(X, aspect="auto", cmap='jet', interpolation='none')
    fig.show()
    fig.savefig('dm.pdf')
    plt.close()

    # %% Get spike train
    X = dm.X
    y = dm.get_binned_spike('sptrain', trial_indices)
    print(f"{y.shape=}")
    assert len(X) == len(y)
    
    # print_sparse(X)
    print(X.shape, y.shape)
    # print_sparse(y)

    # %% Do some processing on the design matrix
    col_inds = dm.get_design_matrix_col_indices('LFP')
    dm.zscore_columns(col_inds)
    
    matfile = read_mat(Path(__file__).parent / "exampleDM.mat")
    matX = matfile['Z']
    # assert dm.X.shape == matX.shape
    
    # for covar in dm.covariates.keys():
    #     col_inds = dm.get_design_matrix_col_indices(covar)

    #     fig, ax = plt.subplots()
    #     D = dm.X[:, col_inds] - matX[:, col_inds]
    #     vm = np.max(np.abs(D))
    #     ax.imshow(D, aspect="auto", cmap="bwr", interpolation='none', vmin=-vm, vmax=vm)
    #     fig.show()
    #     fig.savefig(f"D_{covar}.pdf")
    #     plt.close()        

    #     print(covar, col_inds, np.allclose(dm.X[:, col_inds], matX[:, col_inds]))
    
    assert np.allclose(dm.X, matX)

    # %% Fit GLM
    # glm = PoissonRegressor(alpha=1).fit(dm.X, y)
    glm = PoissonRegressor(alpha=10, solver="newton-cholesky").fit(dm.X, y)
    print(glm.coef_)
    weights = dm.combine_weights(glm.coef_)
    
    n_covars = len(dm.covariates)
    fig, axs = plt.subplots(n_covars, 1, figsize=(12, 2 * n_covars))
    for k, (lk, wk) in enumerate(weights.items()):
        assert lk == wk['label']
        ax = axs[k]
        ax.plot(wk['tr'], wk['data'])
        ax.set_title(lk)
    fig.tight_layout()
    fig.savefig("glm_weights.pdf")
    plt.close()


if __name__ == "__main__":
    main()
