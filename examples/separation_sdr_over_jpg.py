import norbert
import musdb
import numpy as np
import functools
import museval
import pandas as pd


class DF_writer(object):
    def __init__(self, columns):
        self.df = pd.DataFrame(columns=columns)
        self.columns = columns

    def append(self, **row_data):
        if set(self.columns) == set(row_data):
            s = pd.Series(row_data)
            self.df = self.df.append(s, ignore_index=True)

    def df(self):
        return self.df


data = DF_writer(['track', 'SDR', 'percentile', 'size'])


def oracle(track, separation_fn, quantize_mixture=False):
    # set (trackwise) norbert objects
    tf = norbert.TF()
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()
    im = norbert.Coder(format='jpg', quality=90)

    # compute the mixture complex tf transform
    x = tf.transform(track.audio)

    # prepare the spectrograms of the sources
    # get maximum of mixture
    x_c = np.sqrt(
        np.sum(
            np.abs(x)**2,
            axis=-1, keepdims=True
        )
    )

    x_c = ls.scale(x_c)
    mixture_bounds = ls.bounds

    if quantize_mixture:
        x_c = qt.quantize(x_c)
        x_c, file_size = im.encodedecode(x_c)

        x_c = qt.dequantize(x_c)
        x_c = ls.unscale(x_c)

        x = np.multiply(x_c, np.exp(1j * np.angle(x)))
    # bounds = None
    v = []
    for name, value in track.sources.items():
        v_j = np.sqrt(
            np.sum(
                np.abs(tf.transform(value.audio))**2,
                axis=-1, keepdims=True
            )
        )

        v_j = ls.scale(v_j, bounds=mixture_bounds)

        v_j = qt.quantize(v_j)
        v_j, file_size = im.encodedecode(v_j)

        v_j = qt.dequantize(v_j)
        v_j = ls.unscale(v_j, bounds=mixture_bounds)

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

    print(np.mean(scores.means('SDR')))

    data.append(
        track=track.name,
        SDR=np.mean(scores.means('SDR')),
        percentile=0.1,
        size=100
    )

    return estimates


if __name__ == '__main__':
    mus = musdb.DB(is_wav=True)
    tracks = mus.load_mus_tracks(subsets=['test'])
    mus.run(
        functools.partial(
            oracle, separation_fn=norbert.softmask
        ),
        estimates_dir='test_wiener',
        tracks=tracks,
        parallel=False,
        cpus=1
    )

    print(data.df.mean())
