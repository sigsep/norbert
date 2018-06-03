import numpy as np


eps = np.finfo(np.float).eps


class LogScaler(object):
    """apply logarithmic compression and scale to min_max values"""

    def __init__(self):
        self.max = None

    def scale(self, X, max=None):
        # convert to magnitude
        X = np.abs(X)
        # apply log compression
        X_log = np.log(np.maximum(eps, X))

        if max is not None:
            self.max = max
        else:
            self.max = np.max(X_log)

        min = self.min(self.max)
        X_log = np.clip(X_log, min, self.max)

        return (X_log - min) / (self.max - min)

    def unscale(self, X, max=None):
        if max is None:
            min = self.min(self.max)
            return np.exp(X * (self.max - min) + min)
        else:
            min = self.min(max)
            return np.exp(X * (max - min) + min)

    def min(self, max):
        return max - 4*np.log(10)

    def __call__(self, X, forward=True):
        if forward:
            return self.scale(X)
        if not forward:
            return self.unscale(X)
