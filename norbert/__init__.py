import numpy as np
from . transform import TF, Energy
from . bandwidth import BandwidthLimiter
from . scale import LogScaler
from . quantize import Quantizer
from . image import Coder


class Processor(object):
    """docstring for Processor."""
    def __init__(self, modules=[]):
        self.pipeline = modules

    def __getitem__(self, key):
        return self.pipeline[key]

    def __len__(self):
        len(self.pipeline)

    def forward(self, input):
        output = input
        for module in self.pipeline:
            output = module(output, forward=True)
        return output

    def backward(self, input):
        output = input
        for module in reversed(self.pipeline):
            output = module(output, forward=False)
        return output
