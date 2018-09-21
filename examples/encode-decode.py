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
    mono = False
    if mono:
        audio = np.atleast_2d(np.mean(audio, axis=1)).T

    # set up modules
    tf = norbert.TF()
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()
    im = norbert.Coder(format='jpg', quality=85)

    """
    forward path
    """
    # complex spectrogram
    Xc = tf.transform(audio)
    print("Xc", Xc.shape)
    # log scale
    Xs = ls.scale(Xc)
    print("Xs", Xs.shape)
    # quantize to 8bit
    Xq = qt.quantize(Xs)
    print("Xq", Xq.shape)
    # write as jpg image and save bounds values
    im.encode(
        Xq,
        "quantized_image.jpg",
        user_data={'bounds': ls.bounds.tolist()}
    )
    """
    inverse path
    """

    Xm_hat, user_data = im.decode("quantized_image.jpg")
    print(user_data['bounds'])
    print("decode", Xm_hat.shape)
    Xm_hat = qt.dequantize(Xm_hat)
    print("dequantize", Xm_hat.shape)
    Xm_hat = ls.unscale(Xm_hat, bounds=user_data['bounds'])
    print("unscale", Xm_hat.shape)

    # use reconstruction with original phase
    X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
    audio_hat = tf.inverse_transform(X_hat)
    sf.write("reconstruction.wav", audio_hat, rate)
