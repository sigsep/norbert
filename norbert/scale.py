import numpy as np


eps = np.finfo(np.float).eps


class LogScaler(object):
    """apply logarithmic compression and scale to bounds"""

    def scale(self, X, bounds=None):
        # convert to magnitude
        X = np.abs(X)
        # apply log compression
        X_log = np.log(np.maximum(eps, X))

        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = self._bounds(X_log)

        X_log = np.clip(X_log, self.bounds[0], self.bounds[1])

        return (X_log - self.bounds[0]) / (self.bounds[1] - self.bounds[0])

    def unscale(self, X, bounds=None):
        if bounds is None:
            return np.exp(
                X * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            )
        else:
            return np.exp(X * (bounds[1] - bounds[0]) + bounds[0])

    def _bounds(self, X, min=40):
        return np.percentile(X, (min, 100))

    def __call__(self, X, forward=True):
        if forward:
            return self.scale(X)
        if not forward:
            return self.unscale(X)
