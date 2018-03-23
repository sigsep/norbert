import soundfile as sf
import argparse
import wiener as wi
import numpy as np


qtable = [int(s) for s in """
16  11  10  16  24  40  51  61
12  12  14  19  26  58  60  55
14  13  16  24  40  57  69  56
14  17  22  29  51  87  80  62
18  22  37  56  68 109 103  77
24  35  55  64  81 104 113  92
49  64  78  87 103 121 120 101
72  92  95  98 112 100 103  99
""".split(None)]


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
    im = wi.ImageEncoder(format='jpg', quality=25, qtable=qtable)

    # forward path
    # complex spectrogram
    Xc = tf(audio)
    # limit spectrogram to 16Khz
    Xl = bw(Xc)
    #  log scale and quantize spectrogram to 8bit
    Xq = qt(Xl)
    im(Xq, "quantized_image.jpg")

    # backwards
    Xm_hat = im.decode("quantized_image.jpg")
    Xm_hat = qt.dequantize(Xm_hat)
    Xm_hat = bw.upsample(Xm_hat)

    # use reconstruction with original phase
    X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
    audio_hat = tf.inverse_transform(X_hat)
    sf.write("reconstruction.wav", audio_hat, rate)
