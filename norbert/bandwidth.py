import numpy as np


class BandwidthLimiter(object):
    def __init__(self, fs=44100, n_fft=2048, bandwidth=16000):
        self.fs = fs
        self.n_fft = 2048
        self.bandwidth = bandwidth
        self.input_shape = None

    def downsample(self, X):
        self.input_shape = X.shape
        # find bins above `bandwidth`
        ind = np.where(
            np.linspace(
                0,
                float(self.fs) / 2, int(1 + self.n_fft // 2),
                endpoint=True
            ) <= self.bandwidth
        )[0]
        # remove bins above `bandwidth`
        xx = X[X.shape[0] - np.max(ind):, :]

        return xx

    def upsample(self, xx):
        # add zeros for upper part of spectrogram
        XX = np.pad(
            xx,
            ((self.input_shape[0] - xx.shape[0], 0), (0, 0)),
            'constant'
        )
        return XX
