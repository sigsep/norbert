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

        X = tf.transform(track.audio)
        # downmix
        Xi = np.sqrt(np.sum(np.abs(X)**2, axis=-1, keepdims=True))
        Xi = Xi.astype(np.float32)

        track_estimate_dir = os.path.join(
            estimates_dir, track.subset, track.name
        )
        if not os.path.exists(track_estimate_dir):
            os.makedirs(track_estimate_dir)

        # write out tracks to disk

        np.save(os.path.join(track_estimate_dir, 'mix.npy'), Xi)
        for name, track in track.targets.items():
            S = tf.transform(track.audio)
            # downmix
            Si = np.sqrt(np.sum(np.abs(S)**2, axis=-1, keepdims=True))
            Si = Xi.astype(np.float32)

            np.save(os.path.join(track_estimate_dir, name + '.npy'), Si)
