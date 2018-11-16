import numpy as np
import pytest
from norbert.contrib import split, overlapadd


@pytest.fixture(params=[100, 256, 1001])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[1024, 777])
def nb_bins(request):
    return request.param


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[np.float])
def dtype(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def test_len(request, nb_frames):
    return int(nb_frames/request.param)


@pytest.fixture(params=[1, 2, 3])
def test_hop(request, test_len):
    return int(test_len/request.param)


@pytest.fixture
def X(request, nb_frames, nb_bins, nb_channels, dtype):
    np.random.seed(0)
    X = np.random.random((nb_frames, nb_bins, nb_channels)).astype(dtype)
    return X


def test_split(X, test_len, test_hop):
    patches = split(X, test_len, test_hop)
    X_out = overlapadd(patches, 1, test_len, X.shape)
    assert np.allclose(X, X_out)
