# Norbert
Wiener's little toolbox

## Features

* [X] Time-Frequency Transforms
* [X] Log compression
* [X] 8 bit Quantization
* [ ] Generalized Multi-channel Wiener Filter

## Usage Example

```python
import soundfile as sf
import norbert


audio, rate = sf.read(audio_file)

# set up modules
tf = norbert.TF()
bw = norbert.BandwidthLimiter(bandwidth=16000)
ls = norbert.LogScaler()
qt = norbert.Quantizer()
im = norbert.Coder(format='jpg', quality=75)

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
inverse path
"""

Xm_hat = im.decode("quantized_image.jpg")
Xm_hat = qt.dequantize(Xm_hat)
Xm_hat = ls.unscale(Xm_hat)
Xm_hat = bw.upsample(Xm_hat)

# apply reconstruction using original phase
X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
audio_hat = tf.inverse_transform(X_hat)
sf.write("reconstruction.wav", audio_hat, rate)
```
