import argparse
import norbert
import numpy as np
from norbert import eval
import pandas as pd
import musdb
import tqdm


def evaluate(track, mono=True, jpg_quality=75):
    audio = track.audio
    rate = track.rate

    if mono:
        audio = np.atleast_2d(np.mean(audio, axis=1)).T

    # set up modules
    tf = norbert.TF()
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()
    im = norbert.Coder(format='jpg', quality=jpg_quality)

    """
    forward path
    """
    # complex spectrogram
    Xc = tf.transform(audio)
    Xs = ls.scale(Xc)
    Xq = qt.quantize(Xs)
    Y, file_size = im.encodedecode(Xq)
    """
    inverse path
    """

    Xm_hat = qt.dequantize(Y)
    Xm_hat = ls.unscale(Xm_hat)

    # use reconstruction with original phase
    X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
    audio_hat = tf.inverse_transform(X_hat)

    # return peaq score
    return eval.peaqb(audio, audio_hat, rate), file_size


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'musdb_path',
    )

    parser.add_argument(
        '--mono',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()

    mus = musdb.DB(root_dir=args.musdb_path, is_wav=True)
    tracks = mus.load_mus_tracks(subsets='test')

    data = DF_writer(['track', 'quality', 'ODG', 'size'])

    for quality in tqdm.tqdm(range(20, 100)):
        for track in tqdm.tqdm(tracks):
            peaq, file_size = evaluate(
                track, mono=args.mono, jpg_quality=quality
            )
            data.append(
                track=track.name,
                quality=quality,
                ODG=peaq,
                size=file_size
            )

            data.df.to_pickle('results.pickle')
