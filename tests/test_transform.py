import numpy as np
import pytest
import norbert.transform as transform


@pytest.fixture(params=[1024, 2048])
def nb_win(request):
    return request.param


@pytest.fixture(params=[None])
def nb_hop(request):
    return request.param


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2])
def nb_samples(request, rate):
    return int(request.param * rate)


@pytest.fixture(params=[8000, 22050, 44100])
def rate(request):
    return request.param


@pytest.fixture
def x(request, nb_samples, nb_channels):
    return np.random.random((nb_samples, nb_channels))


def test_reconstruction(x, rate, nb_win, nb_hop):
    tf = transform.TF(fs=rate, n_fft=nb_win, n_overlap=nb_hop)

    X = tf.transform(x)
    y = tf.inverse_transform(X)

    assert np.allclose(x, y)


def test_copy(x, rate, nb_win, nb_hop):
    xo = np.copy(x)
    tf = transform.TF(fs=rate, n_fft=nb_win, n_overlap=nb_hop)

    _ = tf.transform(x)
    assert np.allclose(x, xo)
