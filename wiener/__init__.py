import numpy as np
from scipy.signal import stft, istft
import imageio


class TF(object):
    """Time-Frequency transformation aka. spectrogram"""
    def __init__(self, fs=44100, n_fft=2048, overlap=1024):
        self.fs = fs
        self.n_fft = n_fft
        self.overlap = overlap

    def transform(self, x):
        if x.ndim > 1:
            raise ValueError('Only single channel audio!')
        f, t, X = stft(x, nperseg=self.n_fft, noverlap=self.overlap)
        X = np.flipud(X)
        return X

    def inverse_transform(self, X):
        X = np.flipud(X)
        t, audio = istft(X, self.fs, noverlap=self.overlap)
        return audio

    def __call__(self, x):
        return self.transform(x)


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

    def __call__(self, X):
        return self.downsample(X)


class ImageEncoder(object):
    def __init__(self, format='jpg', quality=75, qtable=None):
        self.format = format
        self.quality = quality
        if qtable is not None:
            self.qtables = [qtable]
        else:
            self.qtables = None

    def encode(self, X, out=None):
        if out is not None:
            imageio.imwrite(
                out,
                X,
                format=self.format,
                quality=self.quality,
                optimize=True,
                qtables=self.qtables
            )

    def decode(self, buf):
        img = imageio.imread(buf)
        return np.array(img).astype(np.uint8)

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)


class Quantizer(object):
    """apply log compression and 8bit quantization"""
    def __init__(self, nb_quant=8):
        super(Quantizer, self).__init__()
        self.nb_quant = nb_quant
        self.max = None
        self.min = None
        self.eps = np.finfo(np.float).eps
        self.shape = None

    def quantize(self, X):
        # convert to magnitude
        X = np.abs(X)
        # apply log compression
        X_log = np.log(np.maximum(self.eps, X))
        # save min and max values
        self.min = np.min(X_log)
        self.max = np.max(X_log - self.min)
        # quantize to 8 bit
        Q = (X_log - self.min) / self.max * (
            2**self.nb_quant - 1
        )
        return Q.astype(np.uint8)

    def dequantize(self, Q):
        X_hat = np.exp(
            Q.astype(np.float) / (2 ** self.nb_quant - 1) * self.max + self.min
        )
        return X_hat

    def __call__(self, X):
        return self.quantize(X)
