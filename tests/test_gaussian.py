import numpy as np
import pytest
import norbert.gaussian as gaussian


@pytest.fixture(params=[8, 64, 99])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[8, 64, 99])
def nb_bins(request):
    return request.param


@pytest.fixture(params=[1, 2])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def nb_sources(request):
    return request.param


@pytest.fixture
def X(request, nb_frames, nb_bins, nb_channels):
    return np.random.random(
        (nb_frames, nb_bins, nb_channels)
    ) + np.random.random(
        (nb_frames, nb_bins, nb_channels)
    ) * 1j


@pytest.fixture
def V(request, nb_frames, nb_bins, nb_sources):
    return np.random.random((nb_frames, nb_bins, nb_sources))


def test_shapes(X, V):
    Y = gaussian.wiener(V, X)

    assert X.shape == Y.shape[:-1]

    Y = gaussian.softmask(V, X)

    assert X.shape == Y.shape[:-1]


def test_wiener_copy(X, V):
    X0 = np.copy(X)
    V0 = np.copy(V)

    _ = gaussian.wiener(V, X)

    assert np.allclose(X0, X)
    assert np.allclose(V0, V)


def test_softmask_copy(X, V):
    X0 = np.copy(X)
    V0 = np.copy(V)

    _ = gaussian.softmask(V, X)

    assert np.allclose(X0, X)
    assert np.allclose(V0, V)
