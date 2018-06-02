import numpy as np
import pytest
import norbert.bandwidth as bandwidth


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


@pytest.fixture(params=[np.float, np.complex])
def dtype(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def bandwidth_reduction_factor(request):
    return request.param


@pytest.fixture
def X(request, nb_frames, nb_bins, nb_channels, dtype):
    return np.random.random((nb_frames, nb_bins, nb_channels)).astype(dtype)


def test_reconstruction(X, rate, bandwidth_reduction_factor):
    bw = bandwidth.BandwidthLimiter(
        fs=rate,
        max_bandwidth=float((rate / 2) // bandwidth_reduction_factor)
    )
    Xd = bw.downsample(X)
    assert Xd.shape[1] == X.shape[1] // bandwidth_reduction_factor
    Y = bw.upsample(Xd)
    assert Y.shape == X.shape
