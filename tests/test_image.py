import numpy as np
import pytest
import norbert.image as image
import tempfile as tmp
import os


@pytest.fixture(params=[8, 64, 99])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[8, 64, 99])
def nb_bins(request):
    return request.param


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(
    params=[
        {'a': 1},
        {'a': 2.1},
        {'a': 'test'},
        {'list': ['a', 'b']}
    ]
)
def user_data(request):
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


def test_user_data(X, user_data):
    q = image.Coder(format='jpg')
    image_file = tmp.NamedTemporaryFile(suffix='.jpg')
    q.encode(X, out=image_file.name, user_data=user_data)
    Y, user_data_out = q.decode(image_file.name)
    image_file.close()
    assert user_data == user_data_out
