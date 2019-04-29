# Norbert
:warning: This package is currently work-in-progress

![norbert-wiener](https://user-images.githubusercontent.com/72940/45908695-15ce8900-bdfe-11e8-8420-78ad9bb32f84.jpg) 

[![Build Status](https://travis-ci.org/sigsep/norbert.svg?branch=master)](https://travis-ci.org/sigsep/norbert)
[![Latest Version](https://img.shields.io/pypi/v/norbert.svg)](https://pypi.python.org/pypi/norbert)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/norbert.svg)](https://pypi.python.org/pypi/norbert)

Norbert is an implementation of multichannel Wiener filter, that is a very popular way of filtering multichannel audio for several applications, notably speech enhancement and source separation.

This repository assumes you have some way of estimating power-spectrograms for all the audio sources (non-negative) composing a mixture. It then builds the filter that is appropriate for extracting those signals from a mixture, by optimally exploiting multichannel information (like in stereo signals). This is done in an iterative procedure called _Expectation Maximization_, where filtering and re-estimation of the parameters are iterated.

The current implementation is inspired by three different studies:
1. The actual model is called "Local Gaussian Model" and was introduced in:
  > @article{duong2010under,  
  title={Under-determined reverberant audio source separation using a full-rank spatial covariance model},  
  author={Duong, Ngoc QK and Vincent, Emmanuel and Gribonval, R{\'e}mi},  
  journal={IEEE Transactions on Audio, Speech, and Language Processing},  
  volume={18},  
  number={7},  
  pages={1830--1840},  
  year={2010},  
  publisher={IEEE}  
}  
2. The update method proposed in the original paper was found to sometimes be a bit unstable numerically. For this reason, another modified update was proposed in 
  > @inproceedings{nugraha2016multichannel,  
  title={Multichannel music separation with deep neural networks},  
  author={Nugraha, Aditya Arie and Liutkus, Antoine and Vincent, Emmanuel},  
  booktitle={2016 24th European Signal Processing Conference (EUSIPCO)},  
  pages={1748--1752},  
  year={2016},  
  organization={IEEE}  
}  
3. Finally, an effective trick in practice is to first apply some softmask on the different channels, and use the resulting complex sources as a good initialization for the EM algorithm. This tricy was proposed in:
> @inproceedings{uhlich2017improving,  
  title={Improving music source separation based on deep neural networks through data augmentation and network blending},  
  author={Uhlich, Stefan and Porcu, Marcello and Giron, Franck and Enenkl, Michael and Kemp, Thomas and Takahashi, Naoya and Mitsufuji, Yuki},  
  booktitle={2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  pages={261--265},  
  year={2017},  
  organization={IEEE}  
}  

From a beginner's perspective, all you need to do is to call `norbert.wiener` with the mix and your spectrogram estimates. This should handle the rest.

From a more expert perspective, you will find the different ingredients from the EM algorithm as functions in the module.

## Applications

* Source Separation

## Installation

`pip install norbert`

## Usage

...
