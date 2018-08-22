import norbert
import musdb
import tqdm
import os
import numpy as np

estimates_dir = 'musmag-jpg'

if __name__ == '__main__':
    mus = musdb.DB()
    tracks = mus.load_mus_tracks()
    for track in tqdm.tqdm(tracks):
        # set (trackwise) norbert objects
        tf = norbert.TF()
        ls = norbert.LogScaler()
        qt = norbert.Quantizer()
        im = norbert.Coder(format='jpg', quality=85)

        def pipeline(t, mono=True, bounds=None):
            x = tf.transform(t.audio)
            if mono:
                x = np.sqrt(np.sum(np.abs(x)**2, axis=-1, keepdims=True))

            Q = qt.quantize(
                ls.scale(
                    x,
                    bounds=bounds
                )
            )
            return Q

        # quantize mixture
        Qm = pipeline(track)
        mixture_bounds = ls.bounds
        track_estimate_dir = os.path.join(
            estimates_dir, track.subset, track.name
        )
        if not os.path.exists(track_estimate_dir):
            os.makedirs(track_estimate_dir)

        # write out tracks to disk

        im.encode(Qm, os.path.join(track_estimate_dir, 'mix.jpg'))
        for name, value in track.targets.items():
            Q = pipeline(value, bounds=mixture_bounds)
            im.encode(Q, os.path.join(track_estimate_dir, name + '.jpg'))
