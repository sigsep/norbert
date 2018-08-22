import numpy as np
import itertools


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
        M = M[...]
        M[..., 0, 0] += eps
        M[..., 1, 1] += eps
        # two channels case: analytical expression
        invDet = 1.0/(M[..., 0, 0]*M[..., 1, 1]
                      - M[..., 0, 1]*M[..., 1, 0])
        invM = np.empty_like(M)
        invM[..., 0, 0] = invDet*M[..., 1, 1]
        invM[..., 1, 0] = -invDet*M[..., 1, 0]
        invM[..., 0, 1] = -invDet*M[..., 0, 1]
        invM[..., 1, 1] = invDet*M[..., 0, 0]
    else:
        # general case : no use of analytical expression (slow!)
        invM = np.empyt_like(M)
        for indices in itertools.product(*[range(x) for x in M.shape[:-2]]):
            invM[indices+(Ellipsis,)] = np.linalg.pinv(M[indices+(Ellipsis,)])
    return invM


def posterior(v_j, R_j, inv_Cxx, x, theoretical_cov=False):
    """
    computing the posterior distribution of a source given the mixture

    Parameters
    ----------
    v_j : ndarray, shape (nb_frames, nb_bins, nb_channels, nb_channels).
        power spectral density of the source.
    R_j : ndarray, shape (nb_bins, nb_channels, nb_channels)
        spatial covariance matrix of the source
    inv_Cxx : ndarray, shape (nb_frames, nb_bins, nb_channels, nb_channels)
        inverse of the mixture covariance
    theoretical_cov: boolean
        indicates whether to use the posterior covariance suggested by the
        theory. Defaults to False because this was observed as less
        effective than just the outer product of the posterior mean.

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels)
        posterior average
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_channels)
        posterior covariance
    """
    (nb_bins, nb_channels) = R_j.shape[:2]

    # computes multichannel Wiener gain as v_j R_j inv_Cxx
    G = np.zeros_like(inv_Cxx)
    for (i1, i2, i3) in itertools.product(*(range(nb_channels),)*3):
        G[..., i1, i2] += (v_j[..., i1, i3] * R_j[None, :, i1, i3]
                           * inv_Cxx[..., i3, i2])

    # def logit(W, threshold, slope):
    #     return 1./(1.0+np.exp(-slope*(W-threshold)))
    # regularization = 1e-4
    # thresh = 0.7
    # slope = 30
    # if thresh is not None:
    #     print('applying threshold')
    #     #decomposes into spatial filtering and single sensor Wiener, and
    #     # applies logit compression to the Wiener mask
    #     Wg = np.abs(np.trace(G, axis1=2, axis2=3)) / nb_channels
    #     Wg_new = logit(Wg, thresh, slope)
    #     for i in range(nb_channels):
    #         G[..., i, i] *= (regularization + Wg_new)/(regularization + Wg)
    #     G *= Wg[..., None, None]

    # compute posterior average by (matrix-)multiplying this gain with the mix.
    mu = 0+0j
    for i in range(nb_channels):
        mu += G[..., i] * x[..., i, None]

    # 1/ compute total posterior covariance for source
    C = np.zeros_like(inv_Cxx)
    if theoretical_cov:
        # mu mu'+ (I-G) v_j R_j is the theoretical one, adding the last part
        for (i1, i2, i3) in itertools.product(*(range(nb_channels),)*3):
            C[..., i1, i2] -= G[..., i1, i3] * R_j[None, :, i3, i2]
            C += R_j
        C *= v_j
    for (i1, i2) in itertools.product(*(range(nb_channels),)*2):
        C[..., i1, i2] += mu[..., i1] * np.conj(mu[..., i2])
    return mu, C


def expectation_maximization(v, x, iterations=2):
    """
    expectation maximization, with initial values provided for the sources
    power spectral densities.

    Parameters
    ----------
    v : ndarrays, shape (nb_frames, nb_bins, [nb_channels], nb_sources)
        power spectral densities of the sources. optionally 4D.
        Must be homogeneous to magnitudes, not squared magnitudes !
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        mixture signal
    iterations: int
        number of iterations for the EM algorithm. 1 means processing the
        channels independently with the same filter. More means computing
        spatial statistics.
    update_psd : boolean
        whether or not to also update the provided power spectral densities

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
    eps = np.finfo(np.float).eps*100

    # dimensions
    (nb_frames, nb_bins, nb_channels) = x.shape
    nb_sources = v.shape[-1]

    # define the identity matrx
    identity = np.tile(np.eye(nb_channels, dtype=x.dtype)[None, ...],
                       (nb_bins, 1, 1))

    # initialize the spatial covariance matrices with identity
    # R.shape is (nb_bins, nb_channels, nb_channels, nb_sources)
    R = np.tile(identity[..., None], (1, 1, 1, nb_sources))

    # initialize the results
    y = np.empty((nb_frames, nb_bins, nb_channels, nb_sources), x.dtype)

    print('Number of iterations: ', iterations)
    for it in range(iterations+1):
        # constructing the mixture covariance matrix. Doing it with a loop
        # to avoid storing anytime in RAM the whole 6D tensor
        print('iteration %d' % it)
        print('   inverting mix covariance')

        def v_j(j):
            res = np.zeros((nb_frames, nb_bins, nb_channels, nb_channels))

            def v_ij(i, j):
                return v[..., i, j] if len(v.shape) == 4 else v[..., j]

            for (i1, i2) in itertools.product(*(range(nb_channels),)*2):
                res[..., i1, i2] = v_ij(i1, j) * v_ij(i2, j)
            return res

        Cxx = np.zeros((nb_frames, nb_bins, nb_channels, nb_channels), x.dtype)
        for j in range(nb_sources):
            Cxx += v_j(j) * R[None, ..., j]

        inv_Cxx = invert(Cxx, eps)

        for j in range(nb_sources):
            vj = v_j(j)

            # compute the posterior distribution of the source
            print('   separating source %d' % j)
            y[..., j], C_j = posterior(vj, R[..., j], inv_Cxx, x)

            # for the last iteration, we don't update the parameters
            if it == iterations:
                continue

            # now update the parmeters


            # 1. Udate the power spectral density estimate.
            print('   updating v for source %d' % j)
            v[..., j] = np.sqrt(np.diagonal(np.abs(C_j),
                                            axis1=2, axis2=3))

            # 1. update the spatial covariance matrix
            print('   updating R for source %d' % j)
            #import ipdb; ipdb.set_trace()
            R[..., j] = (np.sum(C_j, axis=0)
                         / (eps + v_j(j).sum(axis=0)))


            # add some regularization to this estimate: normalize and add small
            # identify matrix, so we are sure it behaves well numerically.
            R[..., j] = (R[..., j] * nb_channels / (eps+np.trace(R[..., j, None, None],axis1=1,axis2=2))
                         + 1e-3*identity)

            """Rj_inv = invert(R[..., j], eps)
            for (i1, i2) in itertools.product(*(range(nb_channels),)*2):
                new_v[..., j] += 1./nb_channels*np.real(
                    Rj_inv[None, :, i1, i2] * C_j_post[..., i2, i1]
                )"""

    return y, v, R


def softmask(v, x):
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

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources
    """
    # to avoid dividing by zero
    eps = np.finfo(np.float).eps
    if len(v.shape) == 3:
        v = v[..., None, :]
    total_energy = np.sum(v, axis=-1, keepdims=True)
    return v/(eps + total_energy) * x[..., None]


def wiener(v, x,  update_psd=True, iterations=2):
    """
    apply a multichannel wiener filter to x.
    the spatial statistics are estimated automatically with an EM algorithm,
    using the values provided for the sources PSD as initializion.

    Parameters
    ----------
    v : ndarrays, shape (nb_frames, nb_bins, [nb_channels], nb_sources)
        spectrograms of the sources, homogeneous to magnitudes, optionally 4D.
    x : ndarray, shape (nb_frames, nb_bins, nb_channels)
        mixture signal
    update_psd : boolean
        whether or not to also update the provided power spectral densities
    iterations: int
        number of iterations for the EM algorithm

    Returns
    -------
    ndarray, shape=(nb_frames, nb_bins, nb_channels, nb_sources)
        estimated sources
    """

    return expectation_maximization(v, x, iterations, update_psd)[0]
