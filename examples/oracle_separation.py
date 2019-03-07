import norbert
import musdb
import numpy as np
import functools
import museval


def oracle(track, separation_fn):
    # set (trackwise) norbert objects
    tf = norbert.TF()

    # compute the mixture complex tf transform
    x = tf.transform(track.audio)

    v = []
    for name, value in track.sources.items():
        v_j = np.sum(np.abs(tf.transform(value.audio))**2,
                     axis=-1, keepdims=True)

        v += [np.squeeze(v_j)]

    v = np.moveaxis(np.array(v), 0, 2)

    y = separation_fn(v, x)

    estimates = {}
    for j, (name, value) in enumerate(track.sources.items()):
        audio_hat = tf.inverse_transform(y[..., j])
        estimates[name] = audio_hat

    # Evaluate using museval
    scores = museval.eval_mus_track(
        track, estimates, output_dir=None
    )

    print(scores)

    return estimates


if __name__ == '__main__':
    mus = musdb.DB()
    mus.run(
        functools.partial(
            oracle, separation_fn=norbert.softmask
        ),
        estimates_dir='test_wiener',
        subsets='test',
        parallel=False,
        cpus=1
    )
