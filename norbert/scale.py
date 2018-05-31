import numpy as np


eps = np.finfo(np.float).eps


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
