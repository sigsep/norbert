"""To run this script you need to have peaq installed"""


import subprocess
import re
import numpy as np
import tempfile as tmp
import soundfile as sf


def peaqb(reference, target, rate):
    ref = tmp.NamedTemporaryFile(delete=False, suffix='.wav')
    tgt = tmp.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(ref.name, reference, rate)
    sf.write(tgt.name, target, rate)

    p = subprocess.Popen(
        ['peaqb', '-r', ref.name, '-t', tgt.name],
        stdout=subprocess.PIPE,
        shell=False,
        universal_newlines=True
    )

    ODGs = []
    for line in iter(p.stdout.readline, ''):
        if "ODG" in line:
            values = re.search(r'ODG:\s(.*)', line)
            ODGs.append(float(values.groups()[0]))

    p.stdout.close()

    return np.mean(ODGs)


if __name__ == '__main__':
    A = np.random.random((44100 * 10,))
    B = np.random.random((44100 * 10,))
    print(peaq(A, B, 44100))
