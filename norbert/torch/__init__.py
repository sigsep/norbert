import torch
import math
from .contrib import compress_filter


def expectation_maximization(y, x, iterations=2, verbose=0, eps=None):
    r"""Differentiable expectation maximization algorithm, for refining source separation
    estimates.

    See :func:`norbert.expectation_maximization` for more details.

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
    """Differentiable wiener-based separation for multichannel audio.

    See :func:`norbert.wiener` for more details.
    
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
    """
    See :func:`norbert.softmask` for more details.

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
    Compute the Wiener gain for each source.
    See :func:`norbert.wiener_gain` for more details.


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
    See :func:`norbert.apply_filter` for more details.

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
    See :func:`norbert.get_mix_model` for more details.

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
    See :func:`norbert.get_local_gaussian_model` for more details.

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