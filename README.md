# Norbert

[![Build Status](https://travis-ci.org/sigsep/norbert.svg?branch=master)](https://travis-ci.org/sigsep/norbert)
[![Latest Version](https://img.shields.io/pypi/v/norbert.svg)](https://pypi.python.org/pypi/norbert)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/norbert.svg)](https://pypi.python.org/pypi/norbert)

<img align="left" src="https://user-images.githubusercontent.com/72940/45908695-15ce8900-bdfe-11e8-8420-78ad9bb32f84.jpg">

Norbert is an implementation of multichannel Wiener filter, that is a very popular way of filtering multichannel audio for several applications, notably speech enhancement and source separation.

This repository assumes you have some way of estimating power-spectrograms for all the audio sources (non-negative) composing a mixture. It then builds the filter that is appropriate for extracting those signals from a mixture, by optimally exploiting multichannel information (like in stereo signals). This is done in an iterative procedure called _Expectation Maximization_, where filtering and re-estimation of the parameters are iterated.

From a beginner's perspective, all you need to do is to call `norbert.wiener` with the mix and your spectrogram estimates. This should handle the rest.

From a more expert perspective, you will find the different ingredients from the EM algorithm as functions in the module as described in the [API documentation](https://sigsep.github.io/norbert/)

## Installation

`pip install norbert`

## Usage

...

## License

MIT
