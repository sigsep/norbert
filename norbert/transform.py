import numpy as np
from scipy.signal import stft, istft


class TF(object):
    """Time-Frequency transformation aka. spectrogram"""
    def __init__(self, fs=44100, n_fft=2048, overlap=1024):
        self.fs = fs
        self.n_fft = n_fft
        self.overlap = overlap

    def transform(self, x):
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        f, t, X = stft(x, nperseg=self.n_fft, noverlap=self.overlap)
        X = np.flipud(X)
        return X

    def inverse_transform(self, X):
        X = np.flipud(X)
        t, audio = istft(X, self.fs, noverlap=self.overlap)
        return audio
