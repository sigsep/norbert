import numpy as np
from scipy.ndimage import gaussian_filter


def _logit(W, threshold, slope):
    return 1./(1.0+np.exp(-slope*(W-threshold)))


def residual(v, x, alpha=1):
    """
    adds a model for the residual to the sources models v.
    obtained with simple spectral subtraction after matching the model
    with the mixture as best as possible, frequency wise

    Parameters
    ----------
    v : ndarray, shape (nb_frames, nb_bins, [nb_channels], nb_sources)
        Power spectral densities for the sources, optionally 4-D
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        complex mixture

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_channels)
        Covariance matrix for the mixture
    """
    # to avoid dividing by zero
    eps = np.finfo(np.float).eps

    # making the sources PSD 4-D
    if len(v.shape) == 3:
        v = v[..., None, :]

    # dimensions
    (nb_frames, nb_bins, _, nb_sources) = v.shape

    # spectrogram for the mixture
    vx = np.maximum(eps, np.abs(x)**alpha)

    # compute the total model as provided
    v_total = np.sum(v, axis=-1)

    # quick trick to scale the provided spectrogram to fit the mixture where
    # the model has significant energy
    gain = (
        np.sum(vx*v_total, axis=0, keepdims=True) /
        (eps+np.sum(v_total**2, axis=0, keepdims=True)))
    v *= gain[..., None]
    v_total *= gain

    # residual is difference between the observation and the model
    vr = np.maximum(0, vx - v_total)

    return np.concatenate((v, vr[..., None]), axis=3)


def smooth(v):
    """
    smooth a nonnegative ndarray. Simply apply a small Gaussian blur
    """
    return gaussian_filter(v, sigma=1, truncate=1)


def reduce_interferences(v, thresh=0.6, slope=15):
    """
    reduce interferences between spectrograms with logit compression.
    See [1]_.


    Parameters
    ----------
    v : ndarray, shape=(..., nb_sources)
        non-negative data on which to apply interference reduction
    thresh : float
        threshold for the compression, should be between 0 and 1. The closer
        to 1, the more distortion but the less interferences, hopefully
    slope : float
        the slope at which binarization is done. The higher, the more brutal

    Returns
    -------
    ndarray, Same shape as the filter provided. `v` with reduced interferences

    .. [1] Thomas Prätzlich, Rachel Bittner, Antoine Liutkus, Meinard Müller.
           "Kernel additive modeling for interference reduction in multi-
           channel music recordings" Proc. of ICASSP 2015.

    """
    eps = np.finfo(np.float32).eps
    total_energy = eps + np.sum(v, axis=-1, keepdims=True)
    v = _logit(v/total_energy, 0.4, 15) * v
    return v


def compress_filter(W, eps, thresh=0.6, slope=15, multichannel=True):
    """
    Applies a logit compression to a filter.

    Parameters
    ----------
    W : ndarray, arbitrary shape
        filter on which to apply logit compression. if `multichannel` is False,
        it should be real values between 0 and 1.
    thresh : float
        threshold for the compression, should be between 0 and 1. The closer
        to 1, the more distortion but the less interferences, hopefully
    slope : float
        the slope at which binarization is done. The higher, the more brutal
    multichannel : boolean
        indicate whether we decompose the filter as a beamforming and single
        channel part. In such a case, filter must be of shape
        (nb_frames, nb_bins, [nb_channels, nb_channels]), so either 2D or 4D.
        if it's 2D, it's like multichannel=False. If it's
        4D, the last two dimensions have to be equal.

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, [nb_channels, nb_channels])
        Same shape as the filter provided. Compressed filter
    """
    if W.shape[-1] != W.shape[-2]:
        multichannel = False
    if multichannel:
        if len(W.shape) == 2:
            W = W[..., None, None]
        gains = np.trace(W, axis1=2, axis2=3)
        W *= (_logit(gains, thresh, slope) / (eps + gains))[..., None, None]
    else:
        W = _logit(W, thresh, slope)
    return W
