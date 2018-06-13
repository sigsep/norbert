import numpy as np
from . transform import TF
from . bandwidth import BandwidthLimiter
from . scale import LogScaler
from . quantize import Quantizer
from . image import Coder
from . gaussian import wiener
from . gaussian import softmask


class Processor(object):
    """docstring for Processor."""
    def __init__(self):
        super(Processor, self).__init__()
        # set up modules

        self.tf = TF()
        self.bw = BandwidthLimiter()
        self.ls = LogScaler()
        self.qt = Quantizer()
        self.pipeline = [
            self.tf, self.bw, self.ls, self.qt
        ]

    def forward(self, input):
        output = input
        for module in self.pipeline:
            output = module(output, forward=True)
        return output

    def backward(self, input):
        output = input
        for module in reversed(self.pipeline):
            output = module(output, backward=True)
        return output
