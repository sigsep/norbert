import numpy as np
import pytest
import norbert.image as image


@pytest.fixture(params=[8, 64, 99])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[8, 64, 99])
def nb_bins(request):
    return request.param


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture
def X(request, nb_frames, nb_bins, nb_channels):
    np.random.seed(0)
    return np.random.randint(0, 255, (nb_frames, nb_bins, nb_channels))


def test_shapes(X):
    q = image.Coder(format='png')
    Y, _file_size = q.encodedecode(X)
    assert X.shape == Y.shape


def test_reconstruction(X):
    # use lossless coder to test reconstruction
    q = image.Coder(format='png')
    Y, _file_size = q.encodedecode(X)
    assert X.ndim == Y.ndim
    assert np.allclose(X, Y)
