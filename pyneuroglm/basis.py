import math
from collections import namedtuple
from math import ceil

import numpy as np
from scipy.signal import convolve2d

Basis = namedtuple('Basis', ['func', 'args', 'B', 'edim'])


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

    ttb = np.tile(np.expand_dims(np.arange(1, nbin + 1, dtype=float), 1),
                  (1, nbasis))

    if shape == 'raised cosine':
        dbcenter = nbin / (3. + nbasis)
        width = 4 * dbcenter
        bcenters = 2 * dbcenter + dbcenter * np.arange(nbasis)
        BBstm = rcos(ttb - np.tile(bcenters, (nbin, 1)), width)
    elif shape == 'boxcar':
        width = nbin / nbasis
        BBstm = np.zeros_like(ttb)
        # bcenters = width * np.arange(nbasis) - width / 2
        for k in range(nbasis):
            mask = np.logical_and(ttb[:, k] > ceil(width * k),
                                  ttb[:, k] <= ceil(width * (k + 1)))
            BBstm[mask, k] = 1. / sum(mask)
    else:
        raise ValueError('Unknown shape')

    bases = Basis(func=make_smooth_temporal_basis,
                  args=(shape, duration, nbasis, binfun),
                  B=BBstm,
                  edim=BBstm.shape[1])

    return bases


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
        x = np.concatenate((x, np.zeros((-offset, ndim))), axis=0)
    elif offset > 0:  # causal
        x = np.concatenate((np.zeros((offset, ndim)), x), axis=0)
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
    b = np.asarray(b)
    x = np.zeros((nbin, 1))
    bb = b[b < nbin]
    x[b, :] = v
    return x


def boxcar_stim(start_bin, end_bin, nbin, v=1.):
    x = np.zeros((nbin, 1))
    x[start_bin:end_bin, :] = v
    return x


def _nlin(x, e=1e-20):
    return np.log(x + e)


def _nlinv(x, e=1e-20):
    return np.exp(x) - e


def nonlinear_raised_cosine(x, c, dc):
    return (np.cos(
        np.maximum(-math.pi, np.minimum(math.pi,
                                        (x - c) * math.pi / dc / 2))) + 1) / 2


def make_nonlinear_raised_cosine(nbasis, binsize, endpoints, offset):
    assert offset > 0
    assert len(endpoints) == 2

    endpoints = np.asarray(endpoints)
    yl, yr = _nlin(endpoints + offset)
    db = (yr - yl) / (nbasis - 1)  # what if nbasis = 0?
    # centers = np.arange(yl, yr + db, step=db)  # including endpoint
    # centers = centers[:nbasis]  # make sure centers 
    centers = np.linspace(yl, yr, num=nbasis)
    max_t = _nlinv(yr + 2 * db) - offset
    iht = np.expand_dims(np.arange(0, max_t, step=binsize), -1) / binsize
    
    a = np.tile(_nlin(iht + offset), (1, nbasis))
    b = np.tile(centers, (len(iht), 1))

    ihbasis = nonlinear_raised_cosine(a, b, db)
    # ihctrs = _nlinv(centers)

    bases = Basis(func=make_nonlinear_raised_cosine,
                  args=(nbasis, binsize, endpoints, offset),
                  B=ihbasis,
                  edim=ihbasis.shape[1])

    return bases
