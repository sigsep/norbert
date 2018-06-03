import numpy as np


class Quantizer(object):
    """apply 8-bit quantization"""
    def __init__(self, nb_quant=8):
        super(Quantizer, self).__init__()
        self.nb_quant = nb_quant

    def quantize(self, X):
        # quantize to `nb_quant` bits
        Q = X * (2**self.nb_quant - 1)
        return Q.astype(np.uint8)

    def dequantize(self, Q):
        return Q.astype(np.float) / (2 ** self.nb_quant - 1)

    def __call__(self, X, forward=True):
        if forward:
            return self.quantize(X)
        if not forward:
            return self.dequantize(X)
