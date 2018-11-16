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



import numpy as np
import itertools


def splitinfo(sigShape, frameShape, hop):

    # making sure input shapes are tuples, not simple integers
    if np.isscalar(frameShape):
        frameShape = (frameShape,)
    if np.isscalar(hop):
        hop = (hop,)

    # converting frameShape to array, and building an aligned frameshape,
    # which is 1 whenever the frame dimension is not given. For instance, if
    # frameShape=(1024,) and sigShape=(10000,2), frameShapeAligned is set
    # to (1024,1)
    frameShape = np.array(frameShape)
    fdim = len(frameShape)
    frameShapeAligned = np.append(
        frameShape, np.ones(
            (len(sigShape) - len(frameShape)))).astype(int)

    # same thing for hop
    hop = np.array(hop)
    hop = np.append(hop, np.ones((len(sigShape) - len(hop)))).astype(int)

    # building the positions of the frames. For each dimension, gridding from
    # 0 to sigShape[dim] every hop[dim]
    framesPos = np.ogrid[[slice(0, size, step)
                          for (size, step) in zip(sigShape, hop)]]

    # number of dimensions
    nDim = len(framesPos)

    # now making sure we have at most one frame going out of the signal. This
    # is possible, for instance if the overlap is very large between the frames
    for dim in range(nDim):
        # for each dimension, we remove all frames that go beyond the signal
        framesPos[dim] = framesPos[dim][
            np.nonzero(
                np.add(
                    framesPos[dim],
                    frameShapeAligned[dim]) < sigShape[dim])]
        # are there frames positions left in this dimension ?
        if len(framesPos[dim]):
            # yes. we then add a last frame (the one going beyond the signal),
            # if it is possible. (it may NOT be possible in some exotic cases
            # such as hopSize[dim]>1 and frameShapeAligned[dim]==1
            if framesPos[dim][-1] + hop[dim] < sigShape[dim]:
                framesPos[dim] = np.append(
                    framesPos[dim], framesPos[dim][-1] + hop[dim])
        else:
            # if there is no more frames in this dimension (short signal in
            # this dimension), then at least consider 0
            framesPos[dim] = [0]

    # constructing the shape of the framed signal
    framedShape = np.append(frameShape, [len(x) for x in framesPos])
    return (framesPos, framedShape, frameShape, hop,
            fdim, nDim, frameShapeAligned)


def split(sig, frames_shape, hop, weight_frames=False, verbose=False):
    """splits a ndarray into overlapping frames
    sig : ndarray
    frameShape : tuple giving the size of each frame. If its shape is
                 smaller than that of sig, assume the frame is of size 1
                 for all missing dimensions
    hop : tuple giving the hopsize in each dimension. If its shape is
          smaller than that of sig, assume the hopsize is 1 for all
          missing dimensions
    weightFrames : return frames weighted by a ND hamming window
    verbose : whether to output progress during computation"""

    # signal shape
    sigShape = np.array(sig.shape)

    (framesPos, framedShape, frameShape,
     hop, fdim, nDim, frameShapeAligned) = splitinfo(
                        sigShape, frames_shape, hop)

    if weight_frames:
        # constructing the weighting window. Choosing hamming for convenience
        # (never 0)
        win = 1
        for dim in range(len(frameShape) - 1, -1, -1):
            win = np.outer(np.hamming(frameShapeAligned[dim]), win)
        win = np.squeeze(win)

    # alocating memory for framed signal
    framed = np.zeros(framedShape, dtype=sig.dtype)

    # total number of frames (for displaying)
    nFrames = np.prod([len(x) for x in framesPos])

    # for each frame
    for iframe, index in enumerate(
                itertools.product(*[range(len(x)) for x in framesPos])):
        # display from time to time if asked for
        if verbose and (not iframe % 100):
            print('Splitting : frame ' + str(iframe) + '/' + str(nFrames))

        # build the slice to use for extracting the signal of this frame.
        frameRange = [Ellipsis]
        for dim in range(nDim):
            frameRange += [slice(framesPos[dim][index[dim]],
                                 min(sigShape[dim],
                                     framesPos[dim][index[dim]]
                                     + frameShapeAligned[dim]),
                                 1)]

        # extract the signal
        sigFrame = sig[tuple(frameRange)]
        sigFrame.shape = sigFrame.shape[:fdim]

        # the signal may be shorter than the normal size of a frame (at the
        # end of the signal). We build a slice that corresponds to the actual
        # size we got here
        sigFrameRange = [slice(0, x, 1) for x in sigFrame.shape[:fdim]]

        # puts the signal in the output variable
        framed[tuple(sigFrameRange + list(index))] = sigFrame

        if weight_frames:
            # multiply by the weighting window
            framed[(Ellipsis,) + tuple(index)] *= win

    frameShape = [int(x) for x in frameShape]
    return framed


def overlapadd(S, fdim, hop, shape=None, weighted_frames=True, verbose=False):
    """n-dimensional overlap-add
    S    : ndarray containing the stft to be inverted
    fdim : the number of dimensions in S corresponding to
           frame indices.
    hop  : tuple containing hopsizes along dimensions.
           Missing hopsizes are assumed to be 1
    shape: Indicating the original shape of the
           signal for truncating. If None: no truncating is done
    weightedFrames: True if we need to compensate for the analysis weighting
                    (weightFrames of the split function)
    verbose: whether or not to display progress
            """

    # number of dimensions
    nDim = len(S.shape)

    frameShape = S.shape[:fdim]
    trueFrameShape = np.append(
        frameShape,
        np.ones(
            (nDim - len(frameShape)))).astype(int)

    # same thing for hop
    if np.isscalar(hop):
        hop = (hop,)
    hop = np.array(hop)
    hop = np.append(hop, np.ones((nDim - len(hop)))).astype(int)

    sigShape = [
        (nframedim - 1) * hopdim + frameshapedim for (
            nframedim,
            hopdim,
            frameshapedim) in zip(S.shape[fdim:], hop, trueFrameShape)]

    # building the positions of the frames. For each dimension, gridding from
    # 0 to sigShape[dim] every hop[dim]
    framesPos = [np.arange(size) * step for (size, step)
                 in zip(S.shape[fdim:], hop)]

    # constructing the weighting window. Choosing hamming for convenience
    # (never 0)
    win = np.array(1)
    for dim in range(fdim):
        if trueFrameShape[dim] == 1:
            win = win[..., None]
        else:
            key = ((None,) * len(win.shape) + (Ellipsis,))
            win = (win[..., None]
                   * np.hamming(trueFrameShape[dim]).__getitem__(key))

    # if we need to compensate for analysis weighting, simply square window
    if weighted_frames:
        win2 = win ** 2
    else:
        win2 = win

    sig = np.zeros(sigShape, dtype=S.dtype)

    # will also store the sum of all weighting windows applied during
    # overlap and add. Traditionally, window function and overlap are chosen
    # so that these weights end up being 1 everywhere. However, we here are
    # not restricted here to any particular hopsize. Hence, the price to pay
    # is this further memory burden
    weights = np.zeros(sigShape)

    # total number of frames (for displaying)
    nFrames = np.prod(S.shape[fdim:])

    # could use memmap or stuff
    S *= win[tuple([Ellipsis] + [None] * (len(S.shape) - len(win.shape)))]

    # for each frame
    for iframe, index in enumerate(
            itertools.product(*[range(len(x)) for x in framesPos])):
        # display from time to time if asked for
        if verbose and (not iframe % 100):
            print('overlap-add : frame ' + str(iframe) + '/' + str(nFrames))

        # build the slice to use for overlap-adding the signal of this frame.
        frameRange = [Ellipsis]
        for dim in range(nDim-fdim):
            frameRange += [slice(framesPos[dim][index[dim]],
                                 min(sigShape[dim],
                                     framesPos[dim][index[dim]]
                                     + trueFrameShape[dim]),
                                 1)]

        # put back the reconstructed weighted frame into place
        frameSig = S[tuple([Ellipsis] + list(index))]
        sig[tuple(frameRange)] += frameSig[
                tuple([Ellipsis] +
                      [None] *
                      (len(sig[tuple(frameRange)].shape) -
                      len(frameSig.shape)))]

        # also store the corresponding window contribution
        weights[tuple(frameRange)] += win2[
                tuple([Ellipsis] +
                      [None] *
                      (len(weights[tuple(frameRange)].shape) -
                       len(win2.shape)))]

    # account for different weighting at different places
    sig /= weights

    # truncate the signal if asked for
    if shape is not None:
        sig_res = np.zeros(shape, S.dtype)
        truncateRange = [slice(0, min(x, sig.shape[i]), 1)
                         for (i, x) in enumerate(shape)]
        sig_res[tuple(truncateRange)] = sig[tuple(truncateRange)]
        sig = sig_res

    # finished
    return sig
