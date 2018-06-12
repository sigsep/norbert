import numpy as np
import itertools


def invert(M, eps):
    """
    Inverting matrices

    Parameters
    ----------
    M : ndarray, shape (..., nb_channels, nb_channels)
        matrices to invert: must be square along the last two dimensions

    Returns
    -------
    ndarray, shape=(..., nb_channels, nb_channels)
        invert of M
    """
    nb_channels = M.shape[-1]
    if nb_channels == 1:
        # scalar case
        invM = 1.0/(M+eps)
    elif nb_channels == 2:
        # two channels case: analytical expression
        invDet = 1.0/(eps + M[..., 0, 0]*M[..., 1, 1]
                      - M[..., 0, 1]*M[..., 1, 0])
        invM = np.empty_like(M)
        invM[..., 0, 0] = invDet*M[..., 1, 1]
        invM[..., 1, 0] = -invDet*M[..., 1, 0]
        invM[..., 0, 1] = -invDet*M[..., 0, 1]
        invM[..., 1, 1] = invDet*M[..., 0, 0]
    else:
        # general case : no use of analytical expression (slow!)
        invM = np.empyt_like(M)
        for indices in itertools.product(*[range(x) for x in M.shape[:-2]]):
            invM[indices+(Ellipsis,)] = np.linalg.pinv(M[indices+(Ellipsis,)])
    return invM


def posterior(v_j, R_j, inv_Cxx, x):
    """
    computing the posterior distribution of a source given the mixture

    Parameters
    ----------
    v_j : ndarray, shape (nb_frames, nb_bins)
        power spectral density of the source
    R_j : ndarray, shape (nb_bins, nb_channels, nb_channels)
        spatial covariance matrix of the source
    inv_Cxx : ndarray, shape (nb_frames, nb_bins, nb_channels, nb_channels)
        inverse of the mixture covariance

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels)
        posterior average
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_channels)
        posterior covariance
    """

    (nb_bins, nb_channels) = R_j.shape[:2]

    # computes multichannel Wiener gain as v_j R_j inv_Cxx
    G = np.empty_like(inv_Cxx)
    for (i1, i2, i3) in itertools.product(*(range(nb_channels),)*3):
        G[..., i1, i2] += v_j * R_j[None, :, i1, i3] * inv_Cxx[..., i3, i2]

    # compute posterior average by (matrix-)multiplying this gain with the mix.
    mu = 0
    for i in range(nb_channels):
        mu += G[..., i] * x[..., i, None]

    # 1/ compute observed covariance for source: mu mu'+ (I-G)R_j
    C = np.empty_like(inv_Cxx)
    for (i1, i2, i3) in itertools.product(*(range(nb_channels),)*3):
        C[..., i1, i2] += G[..., i1, i3] * R_j[None, :, i3, i2]
    C += R_j
    for (i1, i2) in itertools.product(*(range(nb_channels),)*2):
        C[..., i1, i2] += mu[..., i1] * np.conj(mu[..., i2])
    return mu, C


def expectation_maximization(v, x, iterations=2, update_psd=True):
    """
    expectation maximization, with initial values provided for the sources
    power spectral densities.

    Parameters
    ----------
    v : ndarrays, shape (nb_frames, nb_bins, nb_sources)
        power spectral densities of the sources. Must be homogeneous to squared
        magnitudes
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        mixture signal
    iterations: int
        number of iterations for the EM algorithm. 1 means processing the
        channels independently with the same filter. More means computing
        spatial statistics.
    update_psd : boolean
        whether or not to also update the provided power spectral densities

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources
    ndarray, shape=(nb_frames, nb_bins, nb_sources)
        estimated power spectral densities
    ndarray, shape=(nb_bins, nb_channels, nb_channels, nb_sources)
        estimated spatial covariance matrices
    """
    # to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape
    nb_sources = v.shape[-1]

    # define the identity matrx
    identity = np.tile(np.eye(nb_channels, dtype = R_j.dtype)[None, ...],
                           (nb_bins, 1, 1)

    # initialize the spatial covariance matrices with identity
    # R.shape is (nb_bins, nb_channels, nb_channels, nb_sources)
    R = np.tile(identity[..., None], (1, 1, 1, nb_sources))

    # initialize the results
    y = np.empty((nb_frames, nb_bins, nb_channels, nb_sources), x.dtype)

    for it in range(iterations+1):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor
        Cxx = np.empty((nb_frames, nb_bins, nb_channels, nb_channels), x.dtype)
        for j in range(nb_sources):
            Cxx += v[..., None, None, j] * R[None, ..., j]

        inv_Cxx = invert(Cxx, eps)

        for j in range(nb_sources):
            # compute the posterior distribution of the source
            y[...,j], C_j = posterior(v[...,j], R[...,j], inv_Cxx)

            # for the last iteration, we don't update the parameters
            if it == iterations:
                continue

            # now update the parmeters

            # 1. update the spatial covariance matrix
            R[..., j] = (np.sum(C_j, axis=0)
                         / (eps+np.sum(v[...,j, None, None], axis=0))

            # add some regularization to this estimate: normalize and add small
            # identify matrix, so we are sure it behaves well numerically.
            R[..., j] = (R[..., j] * nb_channels / np.trace(R[..., j])
                        + eps * identity)

            # 2. Udate the power spectral density estimate.
            if not update_psd:
                continue

            # invert Rj
            Rj_inv = invert(R[..., j], eps)

            # update the PSD
            v[..., j] = 0
            for (i1, i2) in itertools.product(*(range(nb_channels),)*2):
                v[..., j] += 1./nb_channels*np.real(
                    Rj_inv[:, i1, i2][:, None]*C_j[..., i2, i1]
                )

        return y, v, R

def singlechannel_wiener(v, x):
    """
    apply simple ratio mask on all the channels of x, independently,
    using the values provided for the sources spectrograms for
    devising the masks

    Parameters
    ----------
    v : ndarrays, shape (nb_frames, nb_bins, nb_sources)
        spectrograms of the sources
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        mixture signal

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources
    """

    # to avoid dividing by zero
    eps = np.finfo(np.float).eps
    total_energy = np.sum(v, axis=-1)
    return  (v/(eps + total_energy)[..., None])[..., None, :] * x[..., None]

def multichannel_wiener(v, x,  update_psd=True, iterations=2):
    """
    apply a multichannel wiener filter to x.
    the spatial statistics are estimated automatically with an EM algorithm,
    using the values provided for the sources PSD as initializion.

    Parameters
    ----------
    v : ndarrays, shape (nb_frames, nb_bins, nb_sources)
        spectrograms of the sources
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        mixture signal
    update_psd : boolean
        whether or not to also update the provided power spectral densities
    iterations: int
        number of iterations for the EM algorithm

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources
    """

    # to avoid dividing by zero
    return  expectation_maximization(v, x, iterations, update_psd)[0]
