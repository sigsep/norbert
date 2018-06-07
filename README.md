# Norbert

is an audio I/O toolbox for effiently transform, store and filter audio spectrograms, especially suited for machine learning tasks that only rely on non-negative data such as in source separation. In turn, _Norbert_ does use an optimized pipeline to transform and scale audio signals and then apply lossy compression to save them efficiently as JPGs. This makes it an ideal fit to process music data with machine learning libraries such as PyTorch and Tensorflow that have fast, builtin support to load and process images. 

## But Spectrograms are not images, doesn't this destroy audio quality?

__Short answer__: No

__Longer answer:__ It depends on the quanitzation and the use case. We have tested _norbert_ in the context of source separation models where the actual filtering is applied using the original mixture phase, thus reducing the influence of minor imperfections of the magnitude. We used the [PEAQ objective audio quality evaluation](example/jpg_quality_experiment.py) to assess the quality difference in a setting where we  compress the magnitude of and audio signal and synthesize using the decoded (but compressed) magnitude together with the original uncompressed mixture phase. The results on 50 music tracks from the [MUSDB18](sigsep.github.io/musdb18) dataset shows, that with the right JPG quality parameter (we pick `80` as our default), difference between the compressed magnitude and the original magnitude are almost imperceptable. 

![stereo](https://user-images.githubusercontent.com/72940/41040263-2f0a08ba-699c-11e8-9d73-c52e7d04aa25.png)

### File size

Many researchers save their magnitude dataset as numpy pickles or hdf5 files. While this is fast to load and write it uses a significant amount of disk space to store the files (even when zipped). Also, since jpg routines are highly optimized these days, reading jpgs is significantly faster than decoding AAC or MP3 files. 
Here is a small bitrate comparison:

* __npy 64bit:__ ~750 kb/s
* __npy 64bit: zipped:__ ~680 kb/s
* __MP3 good quality:__ 256 kb/s
* __AAC good quality:__ 160 kb/s
* __norbert quantization saved as 8bit npy:__ 89 kb/s
* __norbert quantization saved as 8bit jpg (`q=80`):__ 15 kb/s


## Features

* [X] Time-Frequency Transforms
* [X] Log compression
* [X] 8 bit Quantization
* [ ] Ideal Binary Masks
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

Xm_hat, _ = im.decode("quantized_image.jpg")
Xm_hat = qt.dequantize(Xm_hat)
Xm_hat = ls.unscale(Xm_hat)
Xm_hat = bw.upsample(Xm_hat)

# apply reconstruction using original phase
X_hat = np.multiply(Xm_hat, np.exp(1j * np.angle(Xc)))
audio_hat = tf.inverse_transform(X_hat)
sf.write("reconstruction.wav", audio_hat, rate)
```
