import norbert
import musdb
import numpy as np
import functools
import museval
import scipy


def stft(x, n_fft=2048, n_hopsize=1024):
    _, _, X = scipy.signal.stft(
        x,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        padded=True,
        axis=0
    )
    X = X.transpose((0, 2, 1))
    return X * n_hopsize


def istft(X, rate=44100, n_fft=2048, n_hopsize=1024):
    _, audio = scipy.signal.istft(
        X / n_hopsize,
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        time_axis=1,
        freq_axis=0
    )
    return audio


def oracle(track, separation_fn):
    # compute the mixture complex tf transform
    x = stft(track.audio)
    v = []
    for name, value in track.sources.items():
        v_j = np.sum(np.abs(stft(value.audio))**2,
                     axis=-1, keepdims=True)

        v += [np.squeeze(v_j)]

    v = np.moveaxis(np.array(v), 0, 2)
    y = separation_fn(v, x)

    estimates = {}
    for j, (name, value) in enumerate(track.sources.items()):
        audio_hat = istft(y[..., j])
        estimates[name] = audio_hat

    # Evaluate using museval
    scores = museval.eval_mus_track(
        track, estimates, output_dir=None
    )

    print(scores)

    return estimates


if __name__ == '__main__':
    mus = musdb.DB(download=True)
    mus.run(
        functools.partial(
            oracle, separation_fn=norbert.softmask
        ),
        estimates_dir='test_wiener',
        subsets='test',
        parallel=False,
        cpus=1
    )
