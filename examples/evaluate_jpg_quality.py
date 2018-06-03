import soundfile as sf
import argparse
import norbert
import numpy as np
from norbert import eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
    )
    args = parser.parse_args()
    audio, rate = sf.read(args.input)
    mono = True
    if mono:
        audio = np.atleast_2d(np.mean(audio, axis=1)).T

    # set up modules
    tf = norbert.TF()
    bw = norbert.BandwidthLimiter(max_bandwidth=15000)
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()
    im = norbert.Coder(format='jpg', quality=100)

    """
    forward path
    """
    # complex spectrogram
    Xc = tf.transform(audio)
    Xl = bw.downsample(Xc)
    Xs = ls.scale(Xl)
    Xq = qt.quantize(Xs)
    Y = im.encodedecode(Xq)
    """
    inverse path
    """

    Xm_hat = qt.dequantize(Y)
    Xm_hat = ls.unscale(Xm_hat)
    Xm_hat = bw.upsample(Xm_hat)

    # use reconstruction with original phase
    X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
    audio_hat = tf.inverse_transform(X_hat)

    # print peaq score
    print(eval.peaqb(audio, audio_hat, rate))
