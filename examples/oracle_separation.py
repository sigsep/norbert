import norbert
import musdb
import numpy as np
import functools


def oracle(track, separation_fn):
    # set (trackwise) norbert objects
    tf = norbert.TF()
    bw = norbert.BandwidthLimiter(max_bandwidth=16000)
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()

    # compute the mixture complex tf transform
    x = tf.transform(track. audio)

    # prepare the spectrograms of the sources
    # get maximum of mixture
    ls.scale(bw.downsample(np.sum(np.abs(x)**2, axis=-1)))
    bounds = None
    print(bounds)
    # bounds = None
    v = []
    for name, value in track.sources.items():
        v_j = np.sum(np.abs(tf.transform(value.audio))**2,
                     axis=-1, keepdims=True)

        v_j_Q = qt.quantize(ls.scale(bw.downsample(v_j), bounds))

        v_j_bar = bw.upsample(
            ls.unscale(
                qt.dequantize(v_j_Q),
                bounds
            )
        )

        v += [np.squeeze(v_j_bar)]

    v = np.moveaxis(np.array(v), 0, 2)

    y = separation_fn(v, x)

    estimates = {}
    for j, (name, value) in enumerate(track.sources.items()):
        audio_hat = tf.inverse_transform(y[..., j])
        estimates[name] = audio_hat
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
