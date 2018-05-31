import numpy as np
from scipy.signal import stft, istft
from PIL import Image
import tempfile as tmp
import piexif
import piexif.helper
import json
eps = np.finfo(np.float).eps


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


class ImageEncoder(object):
    def __init__(
        self,
        format='jpg',
        quality=75,
        qtable=None
    ):
        self.format = format
        self.quality = quality

        if qtable is not None:
            self.qtables = [qtable]
        else:
            self.qtables = None

    def encodedecode(self, X):
        """encode/decode on the fly"""
        image_file = tmp.NamedTemporaryFile(suffix='.' + self.format)
        y = self.decode(
            self.encode(X, out=image_file.name)
        )
        image_file.close()
        return y

    def encode(self, X, out=None, user_comment_dict=None):
        if out is not None:
            img = Image.fromarray(X, 'L')

            if user_comment_dict is not None:
                user_comment = piexif.helper.UserComment.dump(
                    json.dumps(user_comment_dict)
                )
                exif_ifd = {
                    piexif.ExifIFD.UserComment: user_comment,
                }
                exif_dict = {"Exif": exif_ifd}
                exif_bytes = piexif.dump(exif_dict)
                img.save(
                    out,
                    quality=self.quality,
                    optimize=True,
                    exif=exif_bytes,
                    qtables=self.qtables
                )
            else:
                img.save(
                    out,
                    quality=self.quality,
                    optimize=True,
                    qtables=self.qtables
                )

    def decode(self, buf):
        img = Image.open(buf)
        return np.array(img).astype(np.uint8)


class LogScaler(object):
    """apply logarithmic compression and scale to min_max values"""

    def __init__(self):
        self.max = None
        self.min = None

    def scale(self, X, min=None, max=None):
        # convert to magnitude
        X = np.abs(X)
        # apply log compression
        X_log = np.log(np.maximum(eps, X))
        if min and max:
            self.min = min
            self.max = max
        else:
            if self.min is None and self.max is None:
                self.self_minmax(X_log)

        X_log = np.clip(X_log, self.min, self.max)

        return (X_log - self.min) / (self.max - self.min)

    def unscale(self, X, min=None, max=None):
        if min is None and max is None:
            return np.exp(X * (self.max - self.min) + self.min)
        else:
            return np.exp(X * (max - min) + min)

    def self_minmax(self, X):
        self.max = np.max(X)
        self.min = self.max - 4*np.log(10)


class Quantizer(object):
    """apply 8bit quantization"""
    def __init__(self, nb_quant=8):
        super(Quantizer, self).__init__()
        self.nb_quant = nb_quant

    def quantize(self, X):
        # quantize to `nb_quant` bits
        Q = X * (2**self.nb_quant - 1)
        return Q.astype(np.uint8)

    def dequantize(self, Q):
        return Q.astype(np.float) / (2 ** self.nb_quant - 1)
