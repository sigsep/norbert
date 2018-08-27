import numpy as np
import itertools
import gaussian


def inverseMixCov(S, R, X=None):
    """
    computes mix inverse covariance matrix
    """
    J = len(S)
    if not J: return None
    Cxx = 0
    for j in range(J):
        Cxx += hat(S[j])[:,:,None,None]*R[j][:,None,...]
    if X is None:
        #normal Wiener separation
        return invert(Cxx)

    #adhoc separation
    (F,T,I) = X.shape

    #mixture "spectrogram"
    Px =np.maximum(regularization,np.mean(np.abs(X)**2,axis=2))

    #computing a mixture covariance matrix where the actual mixture
    #is included when it's too far from the model
    Cxx_temp = Cxx.copy()
    for i in range(I):

        #difference between empirical energy and model
        delta_i = np.maximum(0,Px-Cxx[...,i,i])

        #ratio of this difference over the actual energy, used to weight the energy increase
        W_delta = delta_i/Px

        #creating new mix covariance matrix, including this new energy
        Cxx_temp[...,i,i] += W_delta*delta_i

    Px=0
    delta_i = 0
    W_delta=0

    #inverting this mix covariance matrix
    Cxx_temp = invert(Cxx_temp)

    #separating all target signal
    G = np.zeros(Cxx_temp.shape,dtype='complex64')
    for (i1,i2,i3) in itertools.product(range(I),range(I),range(I)):
        G[...,i1,i2] += Cxx[...,i1,i3]*Cxx_temp[...,i3,i2]
    Cxx=0
    Cxx_temp=0
    Ytarget=0
    for i in range(I):
        Ytarget+=G[...,i]*X[...,i][...,None]
    G=0

    #getting the corresponding residual
    Yresidual = X-Ytarget
    #learn a spatial model for it
    (Rr,Zr) = learnSpatialModel(Yresidual)

    #if available, apply a band pass to it
    if parameters['residualBandpass'] is not None:
        residualBandpass = [min(F,f*F*2/parameters['fs']) for f in parameters['residualBandpass']]
        if residualBandpass[0]>0:
            Zr[:residualBandpass[0],...]=0
        if residualBandpass[1]<F:
            Zr[residualBandpass[1]:,...]=0

    for j in range(J):
        Cxx += hat(S[j])[:,:,None,None]*R[j][:,None,...]
    #finally add this further "source" to the mix covariance matrix model
    Cxx += Zr[:,:,None,None]*Rr[:,None,...]
    return (invert(Cxx), Rr,Zr)
