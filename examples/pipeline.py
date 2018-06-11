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
    en = norbert.Energy()
    bw = norbert.BandwidthLimiter(max_bandwidth=15000)
    ls = norbert.LogScaler()
    qt = norbert.Quantizer()
    im = norbert.Coder(format='jpg', quality=80)

    # use pipeline
    N = norbert.Processor([tf, en, bw, ls, qt])

    # audio to 8bit matrix
    Xq = N.forward(audio)

    # load and write
    im.encode(Xq, "quantized_image.jpg", user_dict={'max': ls.max})
    Xm_hat, user_data = im.decode("quantized_image.jpg")

    # access the complex spectrogram
    print(N[1].X.shape)

    # back to time-domain based on the inverse model
    audio_hat = N.backward(Xm_hat)
    # maybe we don't want to run the full pipeline?
    N.pipeline = N.pipeline[:3]
    audio_hat = N.backward(Xm_hat)
    sf.write("reconstruction.wav", audio_hat, rate)
