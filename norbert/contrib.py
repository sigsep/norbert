import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d


def _logit(W, threshold, slope):
    return 1. / (1.0 + np.exp(-slope*(W-threshold)))


def residual_model(v, x, alpha=1):
    r"""Compute a model for the residual based on spectral subtraction.

    The method consists in two steps:

    * The provided spectrograms are summed up to obtain the *input* model for
      the mixture. This *input* model is scaled frequency-wise to best
      fit with the actual observed mixture spectrogram.

    * The residual model is obtained through spectral subtraction of the
      input model from the mixture spectrogram, with flooring to 0.

    Parameters
    ----------
    v: np.ndarray [shape=(nb_frames, nb_bins, {1, nb_channels}, nb_sources)]
        Estimated spectrograms for the sources

    x: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
        complex mixture

    alpha: float [scalar]
        exponent for the spectrograms `v`. For instance, if `alpha==1`,
        then `v` must be homogoneous to magnitudes, and if `alpha==2`, `v`
        must homogeneous to squared magnitudes.

    Returns
    -------
    v: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_sources+1)]
        Spectrograms of the sources, with an appended one for the residual.

    Note
    ----
    It is not mandatory to input multichannel spectrograms. However, the
    output spectrograms *will* be multichannel.

    Warning
    -------
    You must be careful to set `alpha` as the exponent that corresponds to `v`.
    In other words, *you must have*: ``np.abs(x)**alpha`` homogeneous to `v`.
    """
    # to avoid dividing by zero
    eps = np.finfo(v.dtype).eps

    # spectrogram for the mixture
    vx = np.maximum(eps, np.abs(x)**alpha)

    # compute the total model as provided
    v_total = np.sum(v, axis=-1)

    # quick trick to scale the provided spectrogram to fit the mixture where
    # the model has significant energy
    try:
        gain = (
            np.sum(vx*v_total, axis=0, keepdims=True) /
            (eps+np.sum(v_total**2, axis=0, keepdims=True)))
        v_g = v * gain[..., None]
    except Exception:
        print('Automatic scaling for residual model failed. '
              'This is probably due to a very long file. Trying '
              'without it.')
        v_g = v

    # re-sum the sources to build the new current model
    v_total = np.sum(v, axis=-1)

    # residual is difference between the observation and the model
    vr = np.maximum(0, vx - v_total)

    return np.concatenate((v_g, vr[..., None]), axis=3)


def smooth(v, width=1, temporal=False):
    """
    smoothes a ndarray with a Gaussian blur.

    Parameters
    ----------
    v: np.ndarray [shape=(nb_frames, ...)]
        input array

    sigma: int [scalar]
        lengthscale of the gaussian blur

    temporal: boolean
        if True, will smooth only along time through 1d blur. Will use a
        multidimensional Gaussian blur otherwise.

    Returns
    -------
    result: np.ndarray [shape=(nb_frames, ...)]
        filtered array

    """
    if temporal:
        return gaussian_filter1d(v, sigma=width, axis=0)
    else:
        return gaussian_filter(v, sigma=width, truncate=width)


def reduce_interferences(v, thresh=0.6, slope=15):
    r"""
    Reduction of interferences between spectrograms.

    The objective of the method is to redistribute the energy of the input in
    order to "sparsify" spectrograms along the "source" dimension. This is
    motivated by the fact that sources are somewhat sparse and it is hence
    unlikely that they are all energetic at the same time-frequency bins.

    The method is inspired from [1]_ with ad-hoc modifications.

    References
    ----------

   .. [1] Thomas Prätzlich, Rachel Bittner, Antoine Liutkus, Meinard Müller.
           "Kernel additive modeling for interference reduction in multi-
           channel music recordings" Proc. of ICASSP 2015.

    Parameters
    ----------
    v: np.ndarray [shape=(..., nb_sources)]
        non-negative data on which to apply interference reduction

    thresh: float [scalar]
        threshold for the compression, should be between 0 and 1. The closer
        to 1, the more reduction of the interferences, at the price of more
        distortion.

    slope: float [scalar]
            the slope at which binarization is done. The higher, the more
            brutal

    Returns
    -------
    v: np.ndarray [same shape as input]
        `v` with reduced interferences

    """
    eps = np.finfo(np.float32).eps
    vsmooth = smooth(v, 10)
    total_energy = eps + np.sum(vsmooth, axis=-1, keepdims=True)
    v = _logit(vsmooth/total_energy, 0.4, 15) * v
    return v


def compress_filter(W, thresh=0.6, slope=15):
    '''Applies a logit compression to a filter. This enables to "binarize" a
    separation filter. This allows to reduce interferences at the price
    of distortion.

    In the case of multichannel filters, decomposes them as the cascade of a
    pure beamformer (selection of one direction in space), followed by a
    single-channel mask. Then, compression is applied on the mask only.

    Parameters
    ----------
    W: ndarray, shape=(..., nb_channels, nb_channels)
        filter on which to apply logit compression.

    thresh: float
        threshold for the compression, should be between 0 and 1. The closer
        to 1, the less interferences, but the more distortion.

    slope: float
        the slope at which binarization is done. The higher, the more brutal

    Returns
    -------
    W: np.ndarray [same shape as input]
        Compressed filter
    '''

    eps = np.finfo(W).eps
    nb_channels = W.shape[-1]
    if nb_channels > 1:
        gains = np.trace(W, axis1=-2, axis2=-1)
        W *= (_logit(gains, thresh, slope) / (eps + gains))[..., None, None]
    else:
        W = _logit(W, thresh, slope)
    return W
