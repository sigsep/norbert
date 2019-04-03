import numpy as np
import itertools
from .contrib import residual, reduce_interferences
from .contrib import compress_filter, smooth


def _wiener_gain(v_j, R_j, inv_Cxx):
    """
    compute the wiener gain for separating one source

    Parameters
    ----------
    v_j : ndarray, shape (nb_frames, nb_bins, nb_channels).
        power spectral density of the source.
    R_j : ndarray, shape (nb_bins, nb_channels, nb_channels)
        spatial covariance matrix of the source
    inv_Cxx : ndarray, shape (nb_frames, nb_bins, nb_channels, nb_channels)
        inverse of the mixture covariance
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        mixture on which to apply the Wiener gain.

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels)
        source estimate if x is provided and not None
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_channels)
        wiener filtering matrices
    """
    (nb_bins, nb_channels) = R_j.shape[:2]

    # computes multichannel Wiener gain as v_j R_j inv_Cxx
    G = np.zeros_like(inv_Cxx)
    for (i1, i2, i3) in itertools.product(*(range(nb_channels),)*3):
        G[..., i1, i2] += (R_j[None, :, i1, i3] * inv_Cxx[..., i3, i2])
    G *= v_j[..., None, None]
    return G


def _apply_filter(x, W):
    """
    applies a filter on the mixture

    Parameters
    ----------
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        signal on which to apply the filter.
    W: ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_channels)
        filtering matrices

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels)
        filtered signal
    """
    nb_channels = W.shape[-1]

    # apply the filter
    result = 0+0j
    for i in range(nb_channels):
        result += W[..., i] * x[..., i, None]
    return result


def _identity(nb_bins, nb_channels, nb_sources):
    # define the identity matrx
    identity = np.tile(np.eye(nb_channels, dtype=np.complex64)[None, ...],
                       (nb_bins, 1, 1))
    # initialize the spatial covariance matrices with identity
    # R.shape is (nb_bins, nb_channels, nb_channels, nb_sources)
    R = np.tile(identity[..., None], (1, 1, 1, nb_sources))
    return R


def _get_mix_model(v, R):
    """
    compute the covariance of a mixture based on local Gaussian models.
    simply adds up all the v[..., j] * R[..., j]

    Parameters
    ----------
    v : ndarray, shape (nb_frames, nb_bins, nb_sources)
        Power spectral densities for the sources
    R : ndarray, shape (nb_bins, nb_channels, nb_channels, nb_sources)
          Spatial covariance matrices of each sources

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_channels)
        Covariance matrix for the mixture
    """
    nb_channels = R.shape[1]
    (nb_frames, nb_bins, nb_sources) = v.shape
    Cxx = np.zeros((nb_frames, nb_bins, nb_channels, nb_channels), R.dtype)
    for j in range(nb_sources):
        Cxx += v[..., j, None, None] * R[None, ..., j]
    return Cxx


def _covariance(y_j):
    """
    compute the covariance for a source

    Parameters
    ----------
    y_j : ndarray, shape (nb_frames, nb_bins, nb_channels).
          complex stft of the source.

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_channels)
        just y_j * conj(y_j'): empirical covariance for each TF bin.
    """
    (nb_frames, nb_bins, nb_channels) = y_j.shape
    Cj = np.zeros((nb_frames, nb_bins, nb_channels, nb_channels),
                  y_j.dtype)
    for (i1, i2) in itertools.product(*(range(nb_channels),)*2):
        Cj[..., i1, i2] += y_j[..., i1] * np.conj(y_j[..., i2])

    return Cj


def _get_local_gaussian_model(y_j, eps, smoothing=False, vx=None):
    """
    compute the local Gaussian model for a source. First get the
    PSD, and then the spatial covariance matrix.

    Parameters
    ----------
    y_j : ndarray, shape (nb_frames, nb_bins, nb_channels).
          complex stft of the source.
    eps : float
        regularization term
    smoothing : boolean
        if True, the PSD estimate will be smoothed by a (3x3) Gaussian blur.
    vx : ndarray, broadcastable to (nb_frames, nb_bins)
        If provided and if `smoothing` is True, the smoothed PSD will be
        kept smaller than this, elementwise. The rationale is: the source PSD
        should not be made bigger than the mix.
    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins)
        PSD of the source
    ndarray, shape=(nb_bins, nb_channels, nb_channels)
        Spatial covariance matrix of the source
    """
    v_j = np.mean(np.abs(y_j)**2, axis=2)
    if smoothing:
        v_j = smooth(v_j)
        if vx is not None:
            v_j = np.minimum(v_j, vx)

    # compute the covariance of the source
    C_j = _covariance(y_j)

    # updates the spatial covariance matrix
    R_j = (
        np.sum(C_j, axis=0) /
        (eps+np.sum(v_j[..., None, None], axis=0))
    )
    return v_j, R_j


def expectation_maximization(y, x,
                             iterations=2,
                             smoothing=False):
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
    smoothing: bool
        enable blurring of the PSD estimate during optim, defaults to `False`
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
    eps = np.finfo(np.float32).eps

    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape
    nb_sources = y.shape[-1]

    # allocate the spatial covariance matrices and PSD
    R = np.zeros((nb_bins, nb_channels, nb_channels, nb_sources), x.dtype)
    v = np.zeros((nb_frames, nb_bins, nb_sources))

    if smoothing:
        # if we smooth the PSD, compute the PSD of mix
        vx = np.mean(np.abs(x)**2, axis=2)

    print('Number of iterations: ', iterations)
    for it in range(iterations):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor
        print('EM, iteration %d' % (it+1))

        for j in range(nb_sources):
            # update the spectrogram model for source j
            v[..., j], R[..., j] = _get_local_gaussian_model(
                y[..., j],
                eps,
                smoothing, vx if smoothing else None)

        Cxx = _get_mix_model(v, R)
        inv_Cxx = np.linalg.pinv(Cxx)

        # separate the sources
        for j in range(nb_sources):
            W_j = _wiener_gain(v[..., j], R[..., j], inv_Cxx)
            y[..., j] = _apply_filter(x, W_j)

    return y, v, R


def softmask(v, x, logit=None):
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
    logit: None or float between 0 and 1
        enable a compression of the filter, defaults to `None`. If not None,
        gives the point above which the filter is brought closer to 1, and
        under which it is brought closer to 0.

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources
    """
    # to avoid dividing by zero
    eps = np.finfo(np.float32).eps
    if len(v.shape) == 3:
        v = v[..., None, :]
    total_energy = np.sum(v, axis=-1, keepdims=True)
    filter = v/(eps + total_energy)
    if logit is not None:
        filter = compress_filter(filter, eps, thresh=logit, multichannel=False)
    return filter * x[..., None]


def wiener(v, x, iterations=2, smoothing=False, logit=None):
    """
    apply a multichannel wiener filter to x.
    the spatial statistics are estimated automatically with an EM algorithm,
    using the values provided for the sources PSD as initializion.

    Parameters
    ----------
    v : ndarrays, shape (nb_frames, nb_bins, [nb_channels], nb_sources)
        spectrograms of the sources, optionally 4D.
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        mixture signal
    iterations: int
        number of iterations for the EM algorithm
    smoothing: boolean
        whether or not to slightly smooth the spectrograms during EM
    logit: None or float between 0 and 1
        enable a compression of the initial softmask filter, see the doc for
        softmask.

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources
    """
    y = softmask(v, x, logit)

    if iterations:
        y = expectation_maximization(y, x, iterations, smoothing)[0]
    return y
