import numpy as np
import itertools
from scipy.ndimage import gaussian_filter


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
        M = M[...]
        M[..., 0, 0] += eps
        M[..., 1, 1] += eps
        # two channels case: analytical expression
        invDet = 1.0 / (
            M[..., 0, 0]*M[..., 1, 1] -
            M[..., 0, 1]*M[..., 1, 0]
        )
        invM = np.empty_like(M)
        invM[..., 0, 0] = invDet*M[..., 1, 1]
        invM[..., 1, 0] = -invDet*M[..., 1, 0]
        invM[..., 0, 1] = -invDet*M[..., 0, 1]
        invM[..., 1, 1] = invDet*M[..., 0, 0]
    else:
        # general case : no use of analytical expression (slow!)
        invM = np.empty_like(M)
        for indices in itertools.product(*[range(x) for x in M.shape[:-2]]):
            invM[indices+(Ellipsis,)] = np.linalg.pinv(M[indices+(Ellipsis,)])
    return invM


def separate_one_source(v_j, R_j, inv_Cxx, x):
    """
    compute the wiener gain for separating one source and applies it to the mix

    Parameters
    ----------
    v_j : ndarray, shape (nb_frames, nb_bins, nb_channels).
        power spectral density of the source.
    R_j : ndarray, shape (nb_bins, nb_channels, nb_channels)
        spatial covariance matrix of the source
    inv_Cxx : ndarray, shape (nb_frames, nb_bins, nb_channels, nb_channels)
        inverse of the mixture covariance

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels)
        source estimate
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_channels)
        wiener filtering matrices
    """
    (nb_bins, nb_channels) = R_j.shape[:2]

    # computes multichannel Wiener gain as v_j R_j inv_Cxx
    G = np.zeros_like(inv_Cxx)
    for (i1, i2, i3) in itertools.product(*(range(nb_channels),)*3):
        G[..., i1, i2] += (R_j[None, :, i1, i3] * inv_Cxx[..., i3, i2])
    G *= v_j[..., None, None]

    # compute posterior average by (matrix-)multiplying this gain with the mix.
    mu = 0+0j
    for i in range(nb_channels):
        mu += G[..., i] * x[..., i, None]

    return mu, G


def expectation_maximization(y, x, iterations=2, smoothing=True):
    """
    expectation maximization, with initial values provided for the sources
    power spectral densities.

    Parameters
    ----------
    y : ndarrays, shape (nb_frames, nb_bins, nb_channels, nb_sources)
        Initial estimates for the sources
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        mixture signal
    iterations: int
        number of iterations for the EM algorithm. 1 means processing the
        channels independently with the same filter. More means computing
        spatial statistics.

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
    nb_sources = y.shape[-1]

    # define the identity matrx
    identity = np.tile(np.eye(nb_channels, dtype=x.dtype)[None, ...],
                       (nb_bins, 1, 1))

    # initialize the spatial covariance matrices with identity
    # R.shape is (nb_bins, nb_channels, nb_channels, nb_sources)
    R = np.tile(identity[..., None], (1, 1, 1, nb_sources))

    # initialize the results
    v = np.zeros((nb_frames, nb_bins, nb_sources))

    vx = np.mean(np.abs(x)**2, axis=2)

    print('Number of iterations: ', iterations)
    for it in range(iterations):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor
        print('EM, iteration %d' % (it+1))

        for j in range(nb_sources):
            # update the spectrogram model for source j
            v[..., j] = np.mean(
                                np.abs(y[..., j])**2,
                                axis=2)
            if smoothing:
                v[..., j] = np.minimum(
                                vx,
                                gaussian_filter(
                                    v[..., j],
                                    sigma=1,
                                    truncate=1)
                                )

            # compute the covariance of the source
            Cj = np.zeros((nb_frames, nb_bins, nb_channels, nb_channels),
                          x.dtype)
            for (i1, i2) in itertools.product(*(range(nb_channels),)*2):
                Cj[..., i1, i2] += y[..., i1, j] * np.conj(y[..., i2, j])

            # updates the spatial covariance matrix
            R[..., j] = (
                np.sum(Cj, axis=0) /
                (eps+np.sum(v[..., j, None, None], axis=0))
            )

        Cxx = np.zeros((nb_frames, nb_bins, nb_channels, nb_channels), x.dtype)
        for j in range(nb_sources):
            Cxx += v[..., j, None, None] * R[None, ..., j]
        inv_Cxx = invert(Cxx, eps)

        # separate the sources
        for j in range(nb_sources):
            y[..., j] = separate_one_source(
                            v[..., j], R[..., j], inv_Cxx, x
                        )[0]

    return y, v, R


def softmask(v, x):
    """
    apply simple ratio mask on all the channels of x, independently,
    using the values provided for the sources spectrograms for
    devising the masks

    Parameters
    ----------
    v : ndarray, shape (nb_frames, nb_bins, [nb_channels], nb_sources)
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
    if len(v.shape) == 3:
        v = v[..., None, :]
    total_energy = np.sum(v, axis=-1, keepdims=True)
    return v/(eps + total_energy) * x[..., None]


def wiener(v, x, iterations=2):
    """
    apply a multichannel wiener filter to x.
    the spatial statistics are estimated automatically with an EM algorithm,
    using the values provided for the sources PSD as initializion.

    Parameters
    ----------
    v : ndarrays, shape (nb_frames, nb_bins, [nb_channels], nb_sources)
        spectrograms of the sources, homogeneous to magnitudes, optionally 4D.
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

    y = softmask(v, x)
    if iterations:
        y = expectation_maximization(y, x, iterations)[0]
    return y
