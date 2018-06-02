import numpy as np
import pytest
import norbert.quantize as quantize


@pytest.fixture(params=[100, 1000])
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


@pytest.fixture(params=[4, 8])
def nbits(request):
    return request.param


@pytest.fixture
def X(request, nb_frames, nb_bins, nb_channels):
    np.random.seed(0)
    return np.log(np.random.random((nb_frames, nb_bins, nb_channels)))


def test_shapes(X, nbits):
    q = quantize.Quantizer(nb_quant=nbits)
    Xq = q.quantize(X)
    Y = q.dequantize(Xq)
    assert X.shape == Y.shape


def test_reconstruction(rate, nbits):
    X = np.array([0, 0.4, 0.6, 1])
    q = quantize.Quantizer(nb_quant=nbits)
    Xq = q.quantize(X)
    Y = q.dequantize(Xq)
    assert np.allclose(X, Y)
