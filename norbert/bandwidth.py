import numpy as np


class BandwidthLimiter(object):
    def __init__(self, fs=44100, max_bandwidth=14000):
        """Frequency Domain bandwidth max_bandwidth

        Parameters
        ----------
        fs : int
            sample rate in Hz
        max_bandwidth : float, optional
            reduce the sample rate by given factor
        """

        self.fs = fs
        self.max_bandwidth = max_bandwidth

    def downsample(self, X):
        """
        Apply Downsampling

        Parameters
        ----------
        X : ndarray, shape (nb_frames, nb_bins, nb_channels)
            complex or magnitude STFT output

        Returns
        -------
        Xd : ndarray, shape (nb_frames, new_nb_bins, nb_channels)
            reduced complex or magnitude STFT output
        """
        self.input_shape = X.shape
        # find bins above `bandwidth`
        freqs = np.linspace(
            0,
            float(self.fs) / 2,
            X.shape[1] + 1,
            endpoint=True
        )
        ind = np.where(freqs <= self.max_bandwidth + 1)[0]
        # remove bins above `bandwidth`
        Xd = X[:, :np.max(ind), :]
        return Xd

    def upsample(self, Xd):
        """
        Apply Upsampling by padding with Zeros

        Parameters
        ----------
        Xd : ndarray, shape (nb_frames, nb_bins, nb_channels)
            reduced complex or magnitude STFT output

        Returns
        -------
        X : ndarray, shape (nb_frames, new_nb_bins, nb_channels)
            complex or magnitude STFT output

        """
        # add zeros for upper part of spectrogram
        X = np.pad(
            Xd,
            (
                (0, 0),
                # pad frequency axis (before, after=0)
                (0, self.input_shape[1] - Xd.shape[1]),
                (0, 0)
            ),
            'constant'
        )
        return X
