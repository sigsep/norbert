import numpy as np
import itertools
from .contrib import residual, reduce_interferences
from .contrib import compress_filter, smooth


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
        det = (
            M[..., 0, 0]*M[..., 1, 1] -
            M[..., 0, 1]*M[..., 1, 0])

        invDet = 1.0/(det)
        invM = np.empty_like(M)
        invM[..., 0, 0] = invDet*M[..., 1, 1]
        invM[..., 1, 0] = -invDet*M[..., 1, 0]
        invM[..., 0, 1] = -invDet*M[..., 0, 1]
        invM[..., 1, 1] = invDet*M[..., 0, 0]
    else:
        # general case : no use of analytical expression (slow!)
        invM = np.linalg.pinv(M, eps)
    return invM


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
    (_, nb_channels) = R_j.shape[:2]

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


def _identity(shape, nb_channels):
    # constructs an identity matrix and append the specified shape before
    identity = np.tile(np.eye(nb_channels, dtype=np.complex64),
                       shape+(1, 1))
    return identity


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


def _get_local_gaussian_model(y_j, eps=1.):
    """
    compute the local Gaussian model for a source. First get the
    PSD, and then the spatial covariance matrix.

    Parameters
    ----------
    y_j : ndarray, shape (nb_frames, nb_bins, nb_channels).
          complex stft of the source.
    eps : float
        regularization term
    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins)
        PSD of the source
    ndarray, shape=(nb_bins, nb_channels, nb_channels)
        Spatial covariance matrix of the source
    """
    v_j = np.mean(np.abs(y_j)**2, axis=2)

    # compute the covariance of the source
    C_j = _covariance(y_j)

    # updates the spatial covariance matrix
    # this is where there is a potential overflow problem for very long
    # files, but we ignore this potential issue.
    R_j = (
        np.sum(C_j, axis=0) /
        (eps+np.sum(v_j[..., None, None], axis=0))
    )
    return v_j, R_j


def expectation_maximization(
    y, x, 
    iterations=2, final_smoothing=0, verbose=0, eps=None
):
    """expectation maximization,
    with initial values provided for the sources power spectral densities.

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
    final_smoothing: int
        if > 0, width of the Gaussian temporal blurring to apply to the
        spectrograms before final separation. Introduces interference, but
        reduces distortion.

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
    if eps is None:
        eps = np.finfo(np.real(x).dtype).eps

    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape
    nb_sources = y.shape[-1]

    # allocate the spatial covariance matrices and PSD
    R = np.zeros((nb_bins, nb_channels, nb_channels, nb_sources), x.dtype)
    v = np.zeros((nb_frames, nb_bins, nb_sources))

    if verbose:
        print('Number of iterations: ', iterations)
    regularization = np.sqrt(eps) * (
            _identity((nb_frames, nb_bins), nb_channels))
    for it in range(iterations):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor
        if verbose:
            print('EM, iteration %d' % (it+1))

        for j in range(nb_sources):
            # update the spectrogram model for source j
            v[..., j], R[..., j] = _get_local_gaussian_model(
                y[..., j],
                eps)

        if it == iterations - 1 and final_smoothing > 0:
            if verbose:
                print('  smoothing spectrograms...')
            v = smooth(v, final_smoothing, temporal=True)

        Cxx = _get_mix_model(v, R)
        Cxx += regularization
        inv_Cxx = invert(Cxx, eps)
        # separate the sources
        for j in range(nb_sources):
            W_j = _wiener_gain(v[..., j], R[..., j], inv_Cxx)
            y[..., j] = _apply_filter(x, W_j)

    return y, v, R


def softmask(v, x, logit=None, eps=None):
    """
    apply simple ratio mask on all the channels of x, independently,
    using the values provided for the sources spectrograms for
    devising the masks

    Parameters
    ----------
    v : ndarray, shape (nb_frames, nb_bins, nb_channels, nb_sources)
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
    if eps is None:
        eps = np.finfo(v.dtype).eps
    total_energy = np.sum(v, axis=-1, keepdims=True)
    filter = v/(eps + total_energy)
    if logit is not None:
        filter = compress_filter(filter, eps, thresh=logit, multichannel=False)
    return filter * x[..., None]


def wiener(v, x, iterations=2, use_softmask=1, final_smoothing=0, eps=None):
    """
    apply a multichannel wiener filter to x.
    a first estimate is obtained through soft masking or direct reconstruction,
    and then an iterative EM algorithm is applied to refine the results.

    this notably leads to take the spatial (stereo) statistics of the signals
    into account.

    Parameters
    ----------
    v : ndarrays, shape (nb_frames, nb_bins, nb_channels, nb_sources)
        spectrograms of the sources
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        mixture signal
    iterations: int
        number of iterations for the EM algorithm
    use_softmask: number
        * if zero, then the mixture phase will directly be used with the
        spectrogram for first estimation. Be careful that you need
        __magnitude spectrogram estimates__ in that case (not power
        spectrograms)
        * if nonzero, will use a softmasking strategy instead.
        hint: use zero only if your spectrogram model is pretty good.
    final_smoothing: int
        if > 0, width of the Gaussian temporal blurring to apply to the
        spectrograms before final separation. Introduces interference, but
        reduces distortion.

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources
    """
    if use_softmask:
        # if there's no iteration, don't forget smoothing
        if not iterations and final_smoothing > 0:
            v = smooth(v, final_smoothing)
        y = softmask(v, x, eps=eps)
    else:
        # no smoothing in any case in this setup
        y = v * np.exp(1j*np.angle(x[..., None]))

    if not iterations:
        return y

    # we need to refine the estimates. Scales down the estimates for
    # numerical stability
    max_abs = max(1, np.abs(x).max()/10.)
    x_scaled = x / max_abs
    y = expectation_maximization(y/max_abs, x_scaled, iterations,
                                 final_smoothing, eps=eps)[0]
    return y*max_abs
