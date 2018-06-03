import soundfile as sf
import argparse
import norbert
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
    )
    args = parser.parse_args()
    audio, rate = sf.read(args.input)

    # set up modules
    tf = norbert.TF()
    bw = norbert.BandwidthLimiter(max_bandwidth=15000)
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()
    im = norbert.Coder(format='jpg', quality=75)

    """
    forward path
    """
    # complex spectrogram
    Xc = tf.transform(audio)
    print("Xc", Xc.shape)
    # limit spectrogram to 16Khz
    Xl = bw.downsample(Xc)
    print("Xl", Xl.shape)
    # log scale
    Xs = ls.scale(Xl)
    print("Xs", Xs.shape)
    # quantize to 8bit
    Xq = qt.quantize(Xs)
    print("Xq", Xq.shape)
    # write as jpg image
    im.encode(Xq, "quantized_image.jpg", user_comment_dict={'max': ls.max})

    """
    inverse path
    """

    Xm_hat = im.decode("quantized_image.jpg")
    print("decode", Xm_hat.shape)
    Xm_hat = qt.dequantize(Xm_hat)
    print("dequantize", Xm_hat.shape)
    Xm_hat = ls.unscale(Xm_hat)
    print("unscale", Xm_hat.shape)
    Xm_hat = bw.upsample(Xm_hat)
    print("upsample", Xm_hat.shape)

    # use reconstruction with original phase
    X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
    audio_hat = tf.inverse_transform(X_hat)
    sf.write("reconstruction.wav", audio_hat, rate)
