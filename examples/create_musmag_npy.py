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
        ls = norbert.LogScaler()
        qt = norbert.Quantizer()

        def pipeline(t, mono=False):
            if mono:
                audio = np.sqrt(
                    np.sum(np.abs(t.audio)**2, axis=-1, keepdims=True)
                )
            else:
                audio = t.audio

            Q = qt.quantize(
                ls.scale(
                    tf.transform(audio)
                )
            )
            return Q

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
            import ipdb; ipdb.set_trace()
            np.save(os.path.join(track_estimate_dir, name + '.npy'), S)
