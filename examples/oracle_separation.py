import norbert
import musdb
import numpy as np
import functools
import museval


def oracle(track, separation_fn):
    # set (trackwise) norbert objects
    tf = norbert.TF()
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()
    im = norbert.Coder(format='jpg', quality=80)

    # compute the mixture complex tf transform
    x = tf.transform(track.audio)

    # prepare the spectrograms of the sources
    # get maximum of mixture
    ls.scale(np.sum(np.abs(x)**2, axis=-1))
    # bounds = None
    v = []
    for name, value in track.sources.items():
        v_j = np.sum(np.abs(tf.transform(value.audio))**2,
                     axis=-1, keepdims=True)

        v_j = ls.scale(v_j)
        v_j = qt.quantize(v_j)
        v_j, file_size = im.encodedecode(v_j)

        v_j = qt.dequantize(v_j)
        v_j = ls.unscale(v_j)

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
