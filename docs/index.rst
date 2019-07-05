Welcome to Norbert API Documentation!
=====================================

.. image:: https://user-images.githubusercontent.com/72940/45908695-15ce8900-bdfe-11e8-8420-78ad9bb32f84.jpg
    :alt: norbert_wiener

Norbert is an implementation of the multichannel Wiener filter, that is a very popular way of filtering multichannel audio in the time-frequency domain for several applications, notably speech enhancement and source separation.

This filtering method assumes you have some way of estimating the (nonnegative) spectrograms for all the audio sources composing a mixture. If you only have a model for some *target* sources, and not for the rest, you may use :func:`norbert.contrib.residual_model` to let Norbert create a residual model for you.

Given all source spectrograms and the mixture time-frequency representation, this repository can build and apply the filter that is appropriate for separation, by optimally exploiting multichannel information (like in stereo signals). This is done in an iterative procedure called *Expectation Maximization*, where filtering and re-estimation of the parameters are iterated.

The core functions implemented in Norbert are:

.. autosummary::

   norbert.wiener
   norbert.contrib.residual_model
   norbert.softmask

API documentation
=================

.. automodule:: norbert
    :members:

.. automodule:: norbert.contrib
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Citation
========
