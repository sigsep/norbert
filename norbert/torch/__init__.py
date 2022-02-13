import torch
import math
from .contrib import compress_filter, residual_model


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
    y: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels, nb_sources)]
        initial estimates for the sources

    x: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels)]
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
    y: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels, nb_sources)]
        estimated sources after iterations

    v: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_sources)]
        estimated power spectral densities

    R: torch.Tensor [shape=(batch, nb_bins, nb_channels, nb_channels, nb_sources)]
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
        eps = torch.finfo(x.dtype).eps

    # dimensions
    (batch, nb_frames, nb_bins, nb_channels) = x.shape
    nb_sources = y.shape[-1]

    if verbose:
        print('Number of iterations: ', iterations)
    regularization = math.sqrt(eps) * torch.eye(nb_channels, dtype=x.dtype,
                                                device=x.device)
    for it in range(iterations):
        if verbose:
            print('EM, iteration %d' % (it+1))

        v, R = get_local_gaussian_model(y.transpose(
            3, 4).reshape(batch, nb_frames, -1, nb_channels), eps)
        v, R = v.view(batch, nb_frames, nb_bins, nb_sources), R.view(batch, nb_bins, nb_sources, nb_channels,
                                                                     nb_channels).permute(0, 1, 3, 4, 2)

        Cxx = get_mix_model(v, R).add(regularization)
        inv_Cxx = _invert(Cxx, eps)

        W = wiener_gain(v, R, inv_Cxx)
        y = apply_filter(x, W)

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

    v: torch.Tensor [shape=(batch, nb_frames, nb_bins, {1,nb_channels}, nb_sources)]
        spectrograms of the sources. This is a nonnegative tensor that is
        usually the output of the actual separation method of the user. The
        spectrograms may be mono, but they need to be 4-dimensional in all
        cases.

    x: torch.Tensor [complex, shape=(batch, nb_frames, nb_bins, nb_channels)]
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

    y: torch.Tensor
            [complex, shape=(batch, nb_frames, nb_bins, nb_channels, nb_sources)]
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
        y = softmask(v, x)
    else:
        y = v * torch.exp(1j * torch.angle(x[..., None]))

    if not iterations:
        return y

    # we need to refine the estimates. Scales down the estimates for
    # numerical stability
    max_abs = max(1, x.abs().max() * 0.1)
    y = expectation_maximization(
        y / max_abs, x / max_abs, iterations, eps=eps)[0]
    return y * max_abs


def softmask(v: torch.Tensor, x: torch.Tensor, logit: torch.Tensor = None):
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
    v: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels, nb_sources)]
        spectrograms of the sources

    x: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels)]
        mixture signal

    logit: {None, float between 0 and 1}
        enable a compression of the filter. If not None, it is the threshold
        value for the logit function: a softmask above this threshold is
        brought closer to 1, and a softmask below is brought closer to 0.

    Returns
    -------
    Tensor, shape=(batch, nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources

    """
    # to avoid dividing by zero
    eps = torch.finfo(x.dtype).eps
    total_energy = v.sum(-1, keepdim=True)
    mask = v / (eps + total_energy)
    if logit is not None:
        mask = compress_filter(filter, eps, thresh=logit, multichannel=False)
    return mask * x[..., None]


def _invert(M: torch.Tensor, eps):
    """
    Invert matrices, with special fast handling of the 1x1 and 2x2 cases.

    Will generate errors if the matrices are singular: user must handle this
    through his own regularization schemes.

    Parameters
    ----------
    M: torch.Tensor [shape=(..., nb_channels, nb_channels)]
        matrices to invert: must be square along the last two dimensions

    eps: [scalar]
        regularization parameter to use _only in the case of matrices
        bigger than 2x2

    Returns
    -------
    invM: torch.Tensor, [shape=M.shape]
        inverses of M
    """
    nb_channels = M.shape[-1]
    if nb_channels == 1:
        # scalar case
        invM = M.add(eps).reciprocal()
    elif nb_channels == 2:
        # two channels case: analytical expression
        det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]

        invDet = det.reciprocal()
        invM = torch.empty_like(M)
        invM[..., 0, 0] = invDet * M[..., 1, 1]
        invM[..., 1, 0] = -invDet * M[..., 1, 0]
        invM[..., 0, 1] = -invDet * M[..., 0, 1]
        invM[..., 1, 1] = invDet * M[..., 0, 0]
    else:
        # general case : no use of analytical expression (slow!)
        invM = torch.linalg.pinv(M, eps)
    return invM


def wiener_gain(v_j: torch.Tensor, R_j: torch.Tensor, inv_Cxx: torch.Tensor):
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
    v_j: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_sources)]
        power spectral density of the target source.

    R_j: torch.Tensor [shape=(batch, nb_bins, nb_channels, nb_channels, nb_sources)]
        spatial covariance matrix of the target source

    inv_Cxx: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels, nb_channels)]
        inverse of the mixture covariance matrices

    Returns
    -------

    G: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels, nb_channels, nb_sources)]
        wiener filtering matrices, to apply to the mix, e.g. through
        :func:`apply_filter` to get the target source estimate.

    """
    # computes multichannel Wiener gain as v_j R_j inv_Cxx
    G = torch.einsum('zbcds,znbde->znbces', R_j, inv_Cxx) * \
        v_j[..., None, None, :]
    return G


def apply_filter(x: torch.Tensor, W: torch.Tensor):
    """
    Applies a filter on the mixture. Just corresponds to a matrix
    multiplication.

    Parameters
    ----------
    x: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels)]
        STFT of the signal on which to apply the filter.

    W: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels, nb_channels, nb_sources)]
        filtering matrices, as returned, e.g. by :func:`wiener_gain`

    Returns
    -------
    y_hat: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels)]
        filtered signal
    """
    W = W.permute(0, 1, 2, 5, 3, 4)
    x = x[..., None, :, None]
    y_hat = W @ x
    y_hat = y_hat.squeeze(-1).permute(0, 1, 2, 4, 3)
    return y_hat.contiguous()


def get_mix_model(v: torch.Tensor, R: torch.Tensor):
    """
    Compute the model covariance of a mixture based on local Gaussian models.
    simply adds up all the v[..., j] * R[..., j]

    Parameters
    ----------
    v: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_sources)]
        Power spectral densities for the sources

    R: torch.Tensor [shape=(batch, nb_bins, nb_channels, nb_channels, nb_sources)]
        Spatial covariance matrices of each sources

    Returns
    -------
    Cxx: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels, nb_channels)]
        Covariance matrix for the mixture
    """
    if R.is_complex():
        v = v.to(R.dtype)
    Cxx = torch.einsum('znbs,zbcds->znbcd', v, R)
    return Cxx


def _covariance(y_j: torch.Tensor):
    """
    Compute the empirical covariance for a source.

    Parameters
    ----------
    y_j: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels)].
          complex stft of the source.

    Returns
    -------
    Cj: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels, nb_channels)]
        just y_j * conj(y_j.T): empirical covariance for each TF bin.
    """
    Cj = y_j.unsqueeze(-1) * y_j.unsqueeze(-2).conj()
    return Cj


def get_local_gaussian_model(y_j: torch.Tensor, eps=1.):
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
    y_j: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels)]
          complex stft of the source.
    eps: float [scalar]
        regularization term

    Returns
    -------
    v_j: torch.Tensor [shape=(batch, nb_frames, nb_bins)]
        power spectral density of the source
    R_J: torch.Tensor [shape=(batch, nb_bins, nb_channels, nb_channels)]
        Spatial covariance matrix of the source

    """

    v_j = y_j.abs().pow(2).mean(3)
    weight = v_j.sum(1) + eps
    R_j = _covariance(y_j).sum(1) / weight[..., None, None]
    return v_j, R_j