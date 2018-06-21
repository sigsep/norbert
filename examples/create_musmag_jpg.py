import norbert
import musdb
import tqdm
import os

estimates_dir = 'musmag-jpg'

if __name__ == '__main__':
    mus = musdb.DB()
    tracks = mus.load_mus_tracks()
    for track in tqdm.tqdm(tracks):
        # set (trackwise) norbert objects
        tf = norbert.TF()
        bw = norbert.BandwidthLimiter(bandwidth=16000)
        ls = norbert.LogScaler()
        qt = norbert.Quantizer()
        im = norbert.ImageEncoder(format='jpg', quality=80)

        def pipeline(t):
            Q = qt.quantize(
                ls.scale(
                    bw.downsample(
                        tf.transform(t.audio)
                    )
                )
            )
            return Q

        Qm = pipeline(track)

        track_estimate_dir = os.path.join(
            estimates_dir, track.subset, track.name
        )
        if not os.path.exists(track_estimate_dir):
            os.makedirs(track_estimate_dir)

        # write out tracks to disk

        im.encode(Qm, os.path.join(track_estimate_dir, 'mix.jpg'))
        for name, value in track.targets.items():
            Q = pipeline(value)
            im.encode(Q, os.path.join(track_estimate_dir, name + '.jpg'))
