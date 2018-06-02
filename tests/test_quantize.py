import numpy as np
import pytest
import norbert.quantize as quantize


@pytest.fixture(params=[10, 100])
def nb_frames(request, rate):
    return int(request.param)


@pytest.fixture(params=[1024, 777])
def nb_bins(request):
    return request.param


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[8000, 16000, 44100])
def rate(request):
    return request.param


@pytest.fixture(params=[8])
def nbits(request):
    return request.param


@pytest.fixture
def X(request, nb_frames, nb_bins, nb_channels):
    np.random.seed(0)
    return np.log(np.random.random((nb_frames, nb_bins, nb_channels)))


def test_reconstruction(X, rate, nbits):
    q = quantize.Quantizer(nb_quant=nbits)
    Xq = q.quantize(X)
    Y = q.dequantize(Xq)
    assert Y.shape == X.shape
    assert np.sqrt(((X - Y) ** 2).mean()) < 1e-06
