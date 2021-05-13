from math import ceil

import numpy as np
from scipy.signal import convolve2d


def make_smooth_temporal_basis(shape, duration, nbasis, binfun):
    """

    :param shape: 'raised cosine' or 'boxcar'
    :param duration: time to be covered
    :param nbasis: number of basis
    :param binfun:
    :return:
    """

    def rcos(x, p):
        return (np.abs(x / p) < .5) * (np.cos(x * 2 * np.pi / p) * .5 + .5)

    nbin = binfun(duration)

    ttb = np.tile(np.expand_dims(np.arange(1, nbin + 1, dtype=float), 1), (1, nbasis))

    if shape == 'raised cosine':
        dbcenter = nbin / (3. + nbasis)
        width = 4 * dbcenter
        bcenters = 2 * dbcenter + dbcenter * np.arange(nbasis)
        BBstm = rcos(ttb - np.tile(bcenters, (nbin, 1)), width)
    elif shape == 'boxcar':
        width = nbin / nbasis
        BBstm = np.zeros_like(ttb)
        bcenters = width * np.arange(nbasis) - width / 2
        for k in range(nbasis):
            mask = np.logical_and(ttb[:, k] > ceil(width * k), ttb[:, k] <= ceil(width * (k + 1)))
            BBstm[mask, k] = 1. / sum(mask)
    else:
        raise ValueError('Unknown shape')

    bases = Basis()
    bases.shape = shape
    bases.duration = duration
    bases.nbasis = nbasis
    bases.binfun = binfun
    bases.B = BBstm
    bases.edim = bases.B.shape[1]
    bases.tr = ttb
    bases.centers = bcenters

    return bases


class Basis:
    pass


def temporal_bases(x, bases, mask=None):
    x = np.asarray(x)
    n, ndim = x.shape
    tb, nb = bases.shape
    if mask is None:
        mask = np.full((ndim, nb), fill_value=True, dtype=bool)

    BX = []
    for kcov in range(ndim):
        A = convolve2d(x[:, [kcov]], bases[:, mask[kcov, :]])
        BX.append(A[:n, :])
        # BX[:, k:sI[kcov]] = A[:n, :]
        # k = sI[kcov] + 1
    BX = np.column_stack(BX)
    return BX


def conv_basis(x, bases, offset=0):
    """
    :param x: [T, dx]
    :param bases: basis
    :param offset: scalar
    :return:
    """
    x = np.asarray(x)
    n, ndim = x.shape
    if offset < 0:  # anti-causal
        x = np.column_stack((x, np.zeros((-offset, ndim))))
    elif offset > 0:  # causal
        x = np.column_stack((np.zeros((offset, ndim)), x))
    else:
        pass

    X = temporal_bases(x, bases.B)

    if offset < 0:
        X = X[-offset:, :]
    elif offset > 0:
        X = X[:-offset, :]
    else:
        pass

    return X


def delta_stim(b, nbin, v=1.):
    x = np.zeros((nbin, 1))
    if b < nbin:
        x[b, :] = v
    return x


def boxcar_stim(start_bin, end_bin, nbin, v=1.):
    x = np.zeros((nbin, 1))
    x[start_bin:end_bin, :] = v
    return x
