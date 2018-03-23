#  :hotdog:

## Installation

```
pipenv install
```

## Usage

```python
audio, rate = sf.read(audio)
# convert to mono
audio = audio[:, 0]

# set up modules
tf = wi.TF()
bw = wi.BandwidthLimiter(bandwidth=16000)
qt = wi.Quantizer()
im = wi.ImageEncoder(format='jpg', quality=25, qtable=qtable)

# # forward transforms
# complex spectrogram
Xc = tf(audio)
# limit spectrogram to 16Khz
Xl = bw(Xc)
#  log scale and quantize spectrogram to 8bit
Xq = qt(Xl)
im(Xq, "quantized_image.jpg")

# # backward
Xm_hat = im.decode("quantized_image.jpg")
Xm_hat = qt.dequantize(Xm_hat)
Xm_hat = bw.upsample(Xm_hat)

# use reconstruction with original phase
X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
audio_hat = tf.inverse_transform(X_hat)
sf.write("reconstruction.wav", audio_hat, rate)
```
