import soundfile as sf
import argparse
import wiener as wi
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input',
    )
    args = parser.parse_args()
    audio, rate = sf.read(args.input)
    audio = audio[:, 0]
    sf.write("original.wav", audio, rate)

    tf = wi.TF()
    bw = wi.BandwidthLimiter(bandwidth=16000)
    qt = wi.Quantizer()
    im = wi.ImageEncoder(format='jpg', quality=10)

    # forward path
    # complex spectrogram
    Xc = tf(audio)
    # limit spectrogram to 16Khz
    Xl = bw(Xc)
    #  log scale and quantize spectrogram to 8bit
    Xq = qt(Xl)
    im(Xq, "test.jpg")

    # backwards
    Xm_hat = im.decode("test.jpg")
    Xm_hat = qt.dequantize(Xm_hat)
    Xm_hat = bw.upsample(Xm_hat)

    # use reconstruction with original phase
    X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
    audio_hat = tf.inverse_transform(X_hat)
    sf.write("reconstruction.wav", audio_hat, rate)
