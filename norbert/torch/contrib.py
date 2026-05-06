import torch
import torch.nn.functional as F
from ..numpy.contrib import smooth


def _logit(W, threshold, slope):
    return 1. / (1.0 + torch.exp(-slope * (W - threshold)))


def residual_model(v, x, alpha=1, autoscale=False):
    r"""Compute a model for the residual based on spectral subtraction.

    See :func:`norbert.residual_model` for details.

    Parameters
    ----------
    v: torch.Tensor [shape=(batch, nb_frames, nb_bins, {1, nb_channels}, nb_sources)]
        Estimated spectrograms for the sources

    x: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels)]
        complex mixture

    alpha: float [scalar]
        exponent for the spectrograms `v`. For instance, if `alpha==1`,
        then `v` must be homogoneous to magnitudes, and if `alpha==2`, `v`
        must homogeneous to squared magnitudes.
    autoscale: boolean
        in the case you know that the spectrograms will not have the right
        magnitude, it is important that the models are scaled so that the
        residual is correctly estimated.

    Returns
    -------
    v: torch.Tensor [shape=(batch, nb_frames, nb_bins, nb_channels, nb_sources+1)]
        Spectrograms of the sources, with an appended one for the residual.
    """
    # to avoid dividing by zero
    eps = torch.finfo(v.dtype).eps

    # spectrogram for the mixture
    vx = F.threshold(x.abs() ** alpha, eps, eps)

    # compute the total model as provided
    v_total = v.sum(-1)

    if autoscale:
        # quick trick to scale the provided spectrograms to fit the mixture
        gain = torch.sum(vx * v_total, 1)
        weights = torch.sum(v_total * v_total, 1).add_(eps)
        gain /= weights
        v *= gain[..., None]

        # re-sum the sources to build the new current model
        v_total = v.sum(-1)

    # residual is difference between the observation and the model
    vr = (vx - v_total).relu()

    return torch.cat((v, vr[..., None]), axis=4)


def reduce_interferences(v, thresh=0.6, slope=15):
    r"""
    Reduction of interferences between spectrograms.

    See :func:`norbert.reduce_interferences` for details.

    Parameters
    ----------
    v: torch.Tensor [shape=(..., nb_sources)]
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
    v: torch.Tensor [same shape as input]
        `v` with reduced interferences

    """
    eps = torch.finfo(torch.float32).eps
    vsmooth = smooth(v.detach().cpu().numpy(), 10)
    vsmooth = torch.from_numpy(vsmooth).to(v.device).to(v.dtype)
    total_energy = eps + vsmooth.sum(-1, keepdim=True)
    v = _logit(vsmooth / total_energy, 0.4, 15) * v
    return v


def compress_filter(W, thresh=0.6, slope=15):
    '''Applies a logit compression to a filter. This enables to "binarize" a
    separation filter. This allows to reduce interferences at the price
    of distortion.

    See :func:`norbert.compress_filter` for details.


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
    W: torch.Tensor [same shape as input]
        Compressed filter
    '''

    eps = torch.finfo(W.dtype).eps
    nb_channels = W.shape[-1]
    if nb_channels > 1:
        gains = torch.einsum('...ii', W)
        W *= (_logit(gains, thresh, slope) / (eps + gains))[..., None, None]
    else:
        W = _logit(W, thresh, slope)
    return W
