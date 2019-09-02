# Norbert

[![Build Status](https://travis-ci.com/sigsep/norbert.svg?branch=master)](https://travis-ci.com/sigsep/norbert)
[![Latest Version](https://img.shields.io/pypi/v/norbert.svg)](https://pypi.python.org/pypi/norbert)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/norbert.svg)](https://pypi.python.org/pypi/norbert)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3269749.svg)](https://doi.org/10.5281/zenodo.3269749)

<img align="left" src="https://user-images.githubusercontent.com/72940/45908695-15ce8900-bdfe-11e8-8420-78ad9bb32f84.jpg">

Norbert is an implementation of multichannel Wiener filter, that is a very popular way of filtering multichannel audio for several applications, notably speech enhancement and source separation.

This filtering method assumes you have some way of estimating power or magnitude spectrograms for all the audio sources (non-negative) composing a mixture. If you only have a model for some _target_ sources, and not for the rest, you may use `norbert.residual_model` to let Norbert create a residual model for you.

Given all source spectrograms and the mixture Time-Frequency representation, this repository can build and apply the filter that is appropriate for separation, by optimally exploiting multichannel information (like in stereo signals). This is done in an iterative procedure called _Expectation Maximization_, where filtering and re-estimation of the parameters are iterated.

From a beginner's perspective, all you need to do is often to call `norbert.wiener` with the mix and your spectrogram estimates. This should handle the rest.

From a more expert perspective, you will find the different ingredients from the EM algorithm as functions in the module as described in the [API documentation](https://sigsep.github.io/norbert/)

## Installation

`pip install norbert`

## Usage

Asssuming a complex spectrogram `X`, and a (magnitude) estimate of a target to be extracted from the spectrogram, performing the multichannel wiener filter is as simple as this:

```python
X = stft(audio)
V = model(X)
Y = norbert.wiener(V, X)
estimate = istft(Y)
```

## How to contribute

_norbert_ is a community focused project, we therefore encourage the community to submit bug-fixes and requests for technical support through [github issues](https://github.com/sigsep/norbert/issues/new). For more details of how to contribute, please follow our [`CONTRIBUTING.md`](CONTRIBUTING.md). 

## Authors

Antoine Liutkus, Fabian-Robert Stöter

## Citation

If you want to cite the _Norbert_ software package, please use the DOI from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3269749.svg)](https://doi.org/10.5281/zenodo.3269749)

## License

MIT
