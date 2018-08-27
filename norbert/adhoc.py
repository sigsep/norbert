import numpy as np


def _logit(W, threshold, slope):
    return 1./(1.0+np.exp(-slope*(W-threshold)))


def add_residual_model(v, x, alpha=1):
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
        np.sum(vx*v_total, axis=0, keepdims=True)
        / (eps+np.sum(v_total**2, axis=0, keepdims=True)))
    v *= gain[..., None]
    v_total *= gain

    # residual is difference between the observation and the model
    vr = np.maximum(0, vx - v_total)

    return np.concatenate((v, vr[..., None]), axis=3)
