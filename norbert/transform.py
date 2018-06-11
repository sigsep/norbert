import numpy as np
from scipy.signal import stft, istft


class TF(object):
    """Time-Frequency transformation by short time fourier transform

    Parameters
    ----------
    n_fft : int, optional
        FFT window size, defaults to `1024`
    n_hoverlap : int, optional
        FFT window overlap size, defaults to half of window size
    """
    def __init__(self, fs=44100, n_fft=1024, n_overlap=None):
        self.fs = fs
        self.n_fft = n_fft
        self.n_overlap = n_overlap
        self.input_shape = None

    def transform(self, x):
        """
        Parameters
        ----------
        x : ndarray, shape (nb_samples, nb_channels)
            input audio signal of `ndim = 2`. Use np.atleast_2d() for mono

        Returns
        -------
        ndarray, shape=(nb_frames, nb_bins, nb_channels)
            complex STFT
        """
        self.input_shape = x.shape
        f, t, X = stft(x.T, nperseg=self.n_fft, noverlap=self.n_overlap)
        return X.T

    def inverse_transform(self, X):
        """
        Parameters
        ----------
        X : ndarray, shape (nb_frames, nb_bins, nb_channels)
            complex STFT

        Returns
        -------
        audio: ndarray, shape (nb_samples, nb_channels)
            time domain audio signal
        """

        t, audio = istft(X.T, self.fs, noverlap=self.n_overlap)
        audio = audio.T[:self.input_shape[0], :]
        return audio

    def __call__(self, X, forward=True):
        if forward:
            return self.transform(X)
        if not forward:
            return self.inverse_transform(X)


class Energy(object):
    """Convert to non-negative magnitude"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.X = None

    def magnitude(self, X):
        self.X = X
        return np.abs(X) ** self.alpha

    def complex(self, Xm):
        return np.multiply(Xm, np.exp(1j * np.angle(self.X)))

    def __call__(self, X, forward=True):
        if forward:
            return self.magnitude(X)
        if not forward:
            return self.complex(X)
