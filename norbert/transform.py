import numpy as np
from scipy.signal import stft, istft


class TF(object):
    """Time-Frequency transformation aka. spectrogram"""
    def __init__(self, fs=44100, n_fft=2048, overlap=None):
        self.fs = fs
        self.n_fft = n_fft
        self.overlap = overlap
        self.input_shape = None

    def transform(self, x):
        self.input_shape = x.shape
        f, t, X = stft(x.T, nperseg=self.n_fft, noverlap=self.overlap)
        X = np.flipud(X)
        return X

    def inverse_transform(self, X):
        X = np.flipud(X)
        t, audio = istft(X, self.fs, noverlap=self.overlap)
        audio = audio[:, :self.input_shape[0]]
        return audio.T
