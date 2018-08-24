import norbert
import musdb
import tqdm
import os
import numpy as np

estimates_dir = 'musmag-npy'

if __name__ == '__main__':
    mus = musdb.DB()
    tracks = mus.load_mus_tracks()
    for track in tqdm.tqdm(tracks):
        # set (trackwise) norbert objects
        tf = norbert.TF()

        def pipeline(t, mono=True, bounds=None):
            x = np.abs(tf.transform(t.audio))
            if mono:
                x = np.sqrt(np.sum(np.abs(x)**2, axis=-1, keepdims=True))

            return x.astype(np.float32)

        X = pipeline(track)

        track_estimate_dir = os.path.join(
            estimates_dir, track.subset, track.name
        )
        if not os.path.exists(track_estimate_dir):
            os.makedirs(track_estimate_dir)

        # write out tracks to disk

        np.save(os.path.join(track_estimate_dir, 'mix.npy'), X)
        for name, track in track.targets.items():
            S = pipeline(track)
            np.save(os.path.join(track_estimate_dir, name + '.npy'), S)
