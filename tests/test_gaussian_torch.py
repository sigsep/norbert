import torch
import pytest
import norbert.torch as norbert


@pytest.fixture(params=[8, 11, 33])
def nb_frames(request):
    return int(request.param)


@pytest.fixture(params=[8, 11, 33])
def nb_bins(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def nb_channels(request):
    return request.param


@pytest.fixture(params=[1, 2, 4])
def nb_sources(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def nb_iterations(request):
    return request.param


@pytest.fixture(params=[torch.complex64, torch.complex128])
def dtype(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def batch(request):
    return request.param


@pytest.fixture
def X(request, batch, nb_frames, nb_bins, nb_channels, dtype):
    Mix = torch.randn(batch, nb_frames, nb_bins, nb_channels) + \
        torch.randn(batch, nb_frames, nb_bins, nb_channels) * 1j
    return Mix.to(dtype)


@pytest.fixture
def V(request, batch, nb_frames, nb_bins, nb_channels, nb_sources):
    return torch.rand(batch, nb_frames, nb_bins, nb_channels, nb_sources, requires_grad=True)


def test_shapes(V, X):
    Y = norbert.residual_model(V, X)
    assert X.shape == Y.shape[:-1]

    Y = norbert.wiener(V, X)
    assert X.shape == Y.shape[:-1]

    Y = norbert.softmask(V, X)
    assert X.shape == Y.shape[:-1]


def test_wiener_copy(X, V):
    X0 = X.clone()
    V0 = V.clone()

    _ = norbert.wiener(V, X)

    assert torch.allclose(X0, X)
    assert torch.allclose(V0, V)


def test_softmask_copy(X, V):
    X0 = X.clone()
    V0 = V.clone()

    _ = norbert.softmask(V, X)

    assert torch.allclose(X0, X)
    assert torch.allclose(V0, V)


def test_residual_copy(X, V):
    X0 = X.clone()
    V0 = V.clone()

    _ = norbert.residual_model(V, X)

    assert torch.allclose(X0, X)
    assert torch.allclose(V0, V)


def test_silent_sources(X, V):
    with torch.no_grad():
        V[..., :] = 0.0
    Y = norbert.softmask(V, X)

    assert X.shape == Y.shape[:-1]

    Y = norbert.wiener(V, X)
    assert X.shape == Y.shape[:-1]


def test_softmask(V, X):
    X = (X.shape[-1] * torch.ones(X.shape)).to(torch.complex128)
    Y = norbert.softmask(V, X)
    assert torch.allclose(Y.sum(-1), X)
    Y.sum().backward()


def test_wiener(V, X):
    X = (X.shape[-1] * torch.ones(X.shape)).to(torch.complex128)
    Y = norbert.wiener(V, X)
    assert torch.allclose(Y.sum(-1), X)
    Y.sum().backward()