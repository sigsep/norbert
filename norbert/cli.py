import sys
import argparse
import numpy as np
import soundfile as sf
from . import utils
from . import TF, LogScaler, Quantizer, Coder


def audio2img(inargs=None):
    """
    cli application to conver audio files to spectrogram images
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input',
        type=str,
    )

    parser.add_argument(
        'output',
        type=str
    )

    parser.add_argument(
        '--mono', action='store_true', default=False,
        help='Downmix to mono and write grayscale images',
    )

    parser.add_argument(
        '--nfft', type=int,
        default=2048, help='nfft size in samples, defaults to `2048`'
    )

    parser.add_argument(
        '--hop',
        type=int, default=1024,
        help='hop size in samples, defaults to `1024`'
    )

    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%%(prog)s %s' % utils.__version__
    )

    args = parser.parse_args(inargs)
    audio, rate = sf.read(args.input)

    if args.mono:
        audio = np.atleast_2d(np.mean(audio, axis=1)).T

    # set up modules
    tf = TF(n_fft=args.nfft, n_hopsize=args.hop)
    ls = LogScaler()
    qt = Quantizer()
    im = Coder(format='jpg', quality=85)

    # complex spectrogram
    Xc = tf.transform(audio)
    # log scale
    Xs = ls.scale(Xc)
    # quantize to 8bit
    Xq = qt.quantize(Xs)
    # write as jpg image and save bounds values
    im.encode(
        Xq,
        args.output,
        user_data={'bounds': ls.bounds.tolist()}
    )


if __name__ == '__main__':
    audio2img(sys.argv[1:])
