import numpy as np
import itertools
from .contrib import compress_filter, smooth, residual_model
from .contrib import reduce_interferences


def expectation_maximization(y, x, iterations=2, verbose=0, eps=None):
    r"""Expectation maximization algorithm, for refining source separation
    estimates.

    This algorithm allows to make source separation results better by
    enforcing multichannel consistency for the estimates. This usually means
    a better perceptual quality in terms of spatial artifacts.

    The implementation follows the details presented in [1]_, taking
    inspiration from the original EM algorithm proposed in [2]_ and its
    weighted refinement proposed in [3]_, [4]_.
    It works by iteratively:

     * Re-estimate source parameters (power spectral densities and spatial
       covariance matrices) through :func:`get_local_gaussian_model`.

     * Separate again the mixture with the new parameters by first computing
       the new modelled mixture covariance matrices with :func:`get_mix_model`,
       prepare the Wiener filters through :func:`wiener_gain` and apply them
       with :func:`apply_filter``.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [4] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [5] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Parameters
    ----------
    y: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_sources)]
        initial estimates for the sources

    x: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
        complex STFT of the mixture signal

    iterations: int [scalar]
        number of iterations for the EM algorithm.

    verbose: boolean
        display some information if True

    eps: float or None [scalar]
        The epsilon value to use for regularization and filters.
        If None,  the default will use the epsilon of np.real(x) dtype.

    Returns
    -------
    y: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_sources)]
        estimated sources after iterations

    v: np.ndarray [shape=(nb_frames, nb_bins, nb_sources)]
        estimated power spectral densities

    R: np.ndarray [shape=(nb_bins, nb_channels, nb_channels, nb_sources)]
        estimated spatial covariance matrices


    Note
    -----
        * You need an initial estimate for the sources to apply this
          algorithm. This is precisely what the :func:`wiener` function does.

        * This algorithm *is not* an implementation of the "exact" EM
          proposed in [1]_. In particular, it does compute the posterior
          covariance matrices the same (exact) way. Instead, it uses the
          simplified approximate scheme initially proposed in [5]_ and further
          refined in [3]_, [4]_, that boils down to just take the empirical
          covariance of the recent source estimates, followed by a weighted
          average for the update of the spatial covariance matrix. It has been
          empirically demonstrated that this simplified algorithm is more
          robust for music separation.

    Warning
    -------
        It is *very* important to make sure `x.dtype` is `np.complex`
        if you want double precision, because this function will **not**
        do such conversion for you from `np.complex64`, in case you want the
        smaller RAM usage on purpose.

        It is usually always better in terms of quality to have double
        precision, by e.g. calling :func:`expectation_maximization`
        with ``x.astype(np.complex)``.

        This is notably needed if you let common deep learning frameworks like
        PyTorch or TensorFlow do the STFT, because this usually happens in
        single precision.

    """
    # to avoid dividing by zero
    if eps is None:
        eps = np.finfo(np.real(x[0]).dtype).eps

    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape
    nb_sources = y.shape[-1]

    # allocate the spatial covariance matrices and PSD
    R = np.zeros((nb_bins, nb_channels, nb_channels, nb_sources), x.dtype)
    v = np.zeros((nb_frames, nb_bins, nb_sources))

    if verbose:
        print('Number of iterations: ', iterations)
    regularization = np.sqrt(eps) * (
            np.tile(np.eye(nb_channels, dtype=np.complex64),
                    (nb_frames, nb_bins, 1, 1)))
    for it in range(iterations):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor
        if verbose:
            print('EM, iteration %d' % (it+1))

        for j in range(nb_sources):
            # update the spectrogram model for source j
            v[..., j], R[..., j] = get_local_gaussian_model(
                y[..., j],
                eps)

        Cxx = get_mix_model(v, R)
        Cxx += regularization
        inv_Cxx = _invert(Cxx, eps)
        # separate the sources
        for j in range(nb_sources):
            W_j = wiener_gain(v[..., j], R[..., j], inv_Cxx)
            y[..., j] = apply_filter(x, W_j)

    return y, v, R


def wiener(v, x, iterations=1, use_softmask=True, eps=None):
    """Wiener-based separation for multichannel audio.

    The method uses the (possibly multichannel) spectrograms `v` of the
    sources to separate the (complex) Short Term Fourier Transform `x` of the
    mix. Separation is done in a sequential way by:

    * Getting an initial estimate. This can be done in two ways: either by
      directly using the spectrograms with the mixture phase, or
      by using :func:`softmask`.

    * Refinining these initial estimates through a call to
      :func:`expectation_maximization`.

    This implementation also allows to specify the epsilon value used for
    regularization. It is based on [1]_, [2]_, [3]_, [4]_.

    References
    ----------
    .. [1] S. Uhlich and M. Porcu and F. Giron and M. Enenkl and T. Kemp and
        N. Takahashi and Y. Mitsufuji, "Improving music source separation based
        on deep neural networks through data augmentation and network
        blending." 2017 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP). IEEE, 2017.

    .. [2] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel audio source
        separation with deep neural networks." IEEE/ACM Transactions on Audio,
        Speech, and Language Processing 24.9 (2016): 1652-1664.

    .. [3] A. Nugraha and A. Liutkus and E. Vincent. "Multichannel music
        separation with deep neural networks." 2016 24th European Signal
        Processing Conference (EUSIPCO). IEEE, 2016.

    .. [4] A. Liutkus and R. Badeau and G. Richard "Kernel additive models for
        source separation." IEEE Transactions on Signal Processing
        62.16 (2014): 4298-4310.

    Parameters
    ----------

    v: np.ndarray [shape=(nb_frames, nb_bins, {1,nb_channels}, nb_sources)]
        spectrograms of the sources. This is a nonnegative tensor that is
        usually the output of the actual separation method of the user. The
        spectrograms may be mono, but they need to be 4-dimensional in all
        cases.

    x: np.ndarray [complex, shape=(nb_frames, nb_bins, nb_channels)]
        STFT of the mixture signal.

    iterations: int [scalar]
        number of iterations for the EM algorithm

    use_softmask: boolean
        * if `False`, then the mixture phase will directly be used with the
          spectrogram as initial estimates.

        * if `True`, a softmasking strategy will be used as described in
          :func:`softmask`.

    eps: {None, float}
        Epsilon value to use for computing the separations. This is used
        whenever division with a model energy is performed, i.e. when
        softmasking and when iterating the EM.
        It can be understood as the energy of the additional white noise
        that is taken out when separating.
        If `None`, the default value is taken as `np.finfo(np.real(x[0])).eps`.

    Returns
    -------

    y: np.ndarray
            [complex, shape=(nb_frames, nb_bins, nb_channels, nb_sources)]
        STFT of estimated sources

    Note
    ----

    * Be careful that you need *magnitude spectrogram estimates* for the
      case `softmask==False`.
    * We recommand to use `softmask=False` only if your spectrogram model is
      pretty good, e.g. when the output of a deep neural net. In the case
      it is not so great, opt for an initial softmasking strategy.
    * The epsilon value will have a huge impact on performance. If it's large,
      only the parts of the signal with a significant energy will be kept in
      the sources. This epsilon then directly controls the energy of the
      reconstruction error.

    Warning
    -------
    As in :func:`expectation_maximization`, we recommend converting the
    mixture `x` to double precision `np.complex` *before* calling
    :func:`wiener`.

    """
    if use_softmask:
        y = softmask(v, x, eps=eps)
    else:
        y = v * np.exp(1j*np.angle(x[..., None]))

    if not iterations:
        return y

    # we need to refine the estimates. Scales down the estimates for
    # numerical stability
    max_abs = max(1, np.abs(x).max()/10.)
    x_scaled = x / max_abs
    y = expectation_maximization(y/max_abs, x_scaled, iterations, eps=eps)[0]
    return y*max_abs


def softmask(v, x, logit=None, eps=None):
    """Separates a mixture with a ratio mask, using the provided sources
    spectrograms estimates. Additionally allows compressing the mask with
    a logit function for soft binarization.
    The filter does *not* take multichannel correlations into account.

    The masking strategy can be traced back to the work of N. Wiener in the
    case of *power* spectrograms [1]_. In the case of *fractional* spectrograms
    like magnitude, this filter is often referred to a "ratio mask", and
    has been shown to be the optimal separation procedure under alpha-stable
    assumptions [2]_.

    References
    ----------
    .. [1] N. Wiener,"Extrapolation, Inerpolation, and Smoothing of Stationary
        Time Series." 1949.

    .. [2] A. Liutkus and R. Badeau. "Generalized Wiener filtering with
        fractional power spectrograms." 2015 IEEE International Conference on
        Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2015.

    Parameters
    ----------
    v: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_sources)]
        spectrograms of the sources

    x: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
        mixture signal

    logit: {None, float between 0 and 1}
        enable a compression of the filter. If not None, it is the threshold
        value for the logit function: a softmask above this threshold is
        brought closer to 1, and a softmask below is brought closer to 0.

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources

    """
    # to avoid dividing by zero
    if eps is None:
        eps = np.finfo(np.real(x[0]).dtype).eps
    total_energy = np.sum(v, axis=-1, keepdims=True)
    filter = v/(eps + total_energy.astype(x.dtype))
    if logit is not None:
        filter = compress_filter(filter, eps, thresh=logit, multichannel=False)
    return filter * x[..., None]


def _invert(M, eps):
    """
    Invert matrices, with special fast handling of the 1x1 and 2x2 cases.

    Will generate errors if the matrices are singular: user must handle this
    through his own regularization schemes.

    Parameters
    ----------
    M: np.ndarray [shape=(..., nb_channels, nb_channels)]
        matrices to invert: must be square along the last two dimensions

    eps: [scalar]
        regularization parameter to use _only in the case of matrices
        bigger than 2x2

    Returns
    -------
    invM: np.ndarray, [shape=M.shape]
        inverses of M
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


def wiener_gain(v_j, R_j, inv_Cxx):
    """
    Compute the wiener gain for separating one source, given all parameters.
    It is the matrix applied to the mix to get the posterior mean of the source
    as in [1]_

    References
    ----------
    .. [1] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    Parameters
    ----------
    v_j: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
        power spectral density of the target source.

    R_j: np.ndarray [shape=(nb_bins, nb_channels, nb_channels)]
        spatial covariance matrix of the target source

    inv_Cxx: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_channels)]
        inverse of the mixture covariance matrices

    Returns
    -------

    G: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_channels)]
        wiener filtering matrices, to apply to the mix, e.g. through
        :func:`apply_filter` to get the target source estimate.

    """
    (_, nb_channels) = R_j.shape[:2]

    # computes multichannel Wiener gain as v_j R_j inv_Cxx
    G = np.zeros_like(inv_Cxx)
    for (i1, i2, i3) in itertools.product(*(range(nb_channels),)*3):
        G[..., i1, i2] += (R_j[None, :, i1, i3] * inv_Cxx[..., i3, i2])
    G *= v_j[..., None, None]
    return G


def apply_filter(x, W):
    """
    Applies a filter on the mixture. Just corresponds to a matrix
    multiplication.

    Parameters
    ----------
    x: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
        STFT of the signal on which to apply the filter.

    W: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_channels)]
        filtering matrices, as returned, e.g. by :func:`wiener_gain`

    Returns
    -------
    y_hat: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
        filtered signal
    """
    nb_channels = W.shape[-1]

    # apply the filter
    y_hat = 0+0j
    for i in range(nb_channels):
        y_hat += W[..., i] * x[..., i, None]
    return y_hat


def get_mix_model(v, R):
    """
    Compute the model covariance of a mixture based on local Gaussian models.
    simply adds up all the v[..., j] * R[..., j]

    Parameters
    ----------
    v: np.ndarray [shape=(nb_frames, nb_bins, nb_sources)]
        Power spectral densities for the sources

    R: np.ndarray [shape=(nb_bins, nb_channels, nb_channels, nb_sources)]
        Spatial covariance matrices of each sources

    Returns
    -------
    Cxx: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_channels)]
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
    Compute the empirical covariance for a source.

    Parameters
    ----------
    y_j: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)].
          complex stft of the source.

    Returns
    -------
    Cj: np.ndarray [shape=(nb_frames, nb_bins, nb_channels, nb_channels)]
        just y_j * conj(y_j.T): empirical covariance for each TF bin.
    """
    (nb_frames, nb_bins, nb_channels) = y_j.shape
    Cj = np.zeros((nb_frames, nb_bins, nb_channels, nb_channels),
                  y_j.dtype)
    for (i1, i2) in itertools.product(*(range(nb_channels),)*2):
        Cj[..., i1, i2] += y_j[..., i1] * np.conj(y_j[..., i2])

    return Cj


def get_local_gaussian_model(y_j, eps=1.):
    r"""
    Compute the local Gaussian model [1]_ for a source given the complex STFT.
    First get the power spectral densities, and then the spatial covariance
    matrix, as done in [1]_, [2]_

    References
    ----------
    .. [1] N.Q. Duong and E. Vincent and R.Gribonval. "Under-determined
        reverberant audio source separation using a full-rank spatial
        covariance model." IEEE Transactions on Audio, Speech, and Language
        Processing 18.7 (2010): 1830-1840.

    .. [2] A. Liutkus and R. Badeau and G. Richard. "Low bitrate informed
        source separation of realistic mixtures." 2013 IEEE International
        Conference on Acoustics, Speech and Signal Processing. IEEE, 2013.

    Parameters
    ----------
    y_j: np.ndarray [shape=(nb_frames, nb_bins, nb_channels)]
          complex stft of the source.
    eps: float [scalar]
        regularization term

    Returns
    -------
    v_j: np.ndarray [shape=(nb_frames, nb_bins)]
        power spectral density of the source
    R_J: np.ndarray [shape=(nb_bins, nb_channels, nb_channels)]
        Spatial covariance matrix of the source

    """

    v_j = np.mean(np.abs(y_j)**2, axis=2)

    # compute the covariance of the source
    C_j = _covariance(y_j)

    # updates the spatial covariance matrix
    # this is where there is a potential overflow problem for very long
    # files, but we ignore this potential issue and let it to the user.
    try:
        R_j = (
            np.sum(C_j, axis=0) /
            (eps+np.sum(v_j[..., None, None], axis=0))
        )
    except Exception:
        raise Exception(
              'There was a problem when computing the spatial covariance '
              'matrix. Your mixture is probably too long for this '
              'implementation')
    return v_j, R_j
