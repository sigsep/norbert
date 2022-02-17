from typing import Iterable, Tuple
import numpy as np
import torch
import norbert.numpy as norbert_np
import norbert.torch as norbert_torch


nb_frames = 33
nb_bins = 33
nb_channels = 2
nb_sources = 4
nb_iterations = 1
eps = 1e-12


def get_random(*shape):
    return np.random.randn(*shape)


def get_random_complex(*shape):
    return get_random(*shape) + 1j * get_random(*shape)


def get_X_V():
    return get_random_complex(nb_frames, nb_bins, nb_channels), np.abs(get_random(nb_frames, nb_bins, nb_channels, nb_sources))


def add_batch_dim(*args):
    ret = []
    for x in args:
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).unsqueeze(0)
        ret.append(x)

    return tuple(ret)


def assert_equal(y_np, y_torch):
    if not isinstance(y_np, Tuple):
        y_np, y_torch = (y_np,), (y_torch,)

    for n, t in zip(y_np, y_torch):
        assert np.allclose(n, t.squeeze(0).numpy(), atol=1e-16)


def test_softmask():
    X, V = get_X_V()
    np_result = norbert_np.softmask(V, X, eps=eps)
    torch_result = norbert_torch.softmask(*add_batch_dim(V, X), eps=eps)
    assert_equal(np_result, torch_result)


def test_wiener_gain():
    v_j = np.abs(get_random(nb_frames, nb_bins, nb_sources))
    R_j = get_random_complex(nb_bins, nb_channels, nb_channels, nb_sources)
    inv_Cxx = get_random_complex(nb_frames, nb_bins, nb_channels, nb_channels)

    np_result = []
    for i in range(nb_sources):
        np_result.append(
            norbert_np.wiener_gain(v_j[..., i], R_j[..., i], inv_Cxx)
        )
    np_result = np.stack(np_result, -1)
    torch_result = norbert_torch.wiener_gain(*add_batch_dim(v_j, R_j, inv_Cxx))
    assert_equal(np_result, torch_result)


def test_apply_filter():
    X, _ = get_X_V()
    W = get_random_complex(
        nb_frames, nb_bins, nb_channels, nb_channels, nb_sources)

    np_result = []
    for i in range(nb_sources):
        np_result.append(
            norbert_np.apply_filter(X, W[..., i])
        )
    np_result = np.stack(np_result, -1)
    torch_result = norbert_torch.apply_filter(*add_batch_dim(X, W))
    assert_equal(np_result, torch_result)


def test_get_mix_model():
    V = np.abs(get_random(nb_frames, nb_bins, nb_sources))
    R = get_random_complex(nb_bins, nb_channels, nb_channels, nb_sources)

    np_result = norbert_np.get_mix_model(V, R)
    torch_result = norbert_torch.get_mix_model(*add_batch_dim(V, R))
    assert_equal(np_result, torch_result)


def test_get_local_gaussian_model():
    V = get_random_complex(nb_frames, nb_bins, nb_sources)

    np_result = norbert_np.get_local_gaussian_model(V)
    torch_result = norbert_torch.get_local_gaussian_model(*add_batch_dim(V))
    assert_equal(np_result, torch_result)
