import soundfile as sf
import argparse
import norbert
import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
    )
    args = parser.parse_args()
    audio, rate = sf.read(args.input)

    # set up modules
    tf = norbert.TF()
    bw = norbert.BandwidthLimiter(bandwidth=16000)
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()
    im = norbert.ImageEncoder(format='jpg', quality=75)

    """
    forward path
    """
    # complex spectrogram
    Xc = tf.transform(audio)
    # limit spectrogram to 16Khz
    Xl = bw.downsample(Xc)
    # log scale
    Xs = ls.scale(Xl)
    # quantize to 8bit
    Xq = qt.quantize(Xs)
    # write as jpg image
    im.encode(Xq, "quantized_image.jpg")

    """
    forward path
    """

    Xm_hat = im.decode("quantized_image.jpg")
    Xm_hat = qt.dequantize(Xm_hat)
    Xm_hat = ls.unscale(Xm_hat)
    Xm_hat = bw.upsample(Xm_hat)

    # use reconstruction with original phase
    X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
    audio_hat = tf.inverse_transform(X_hat)
    sf.write("reconstruction.wav", audio_hat, rate)
