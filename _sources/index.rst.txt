Welcome to Norbert API Documentation!
=====================================

.. image:: https://user-images.githubusercontent.com/72940/45908695-15ce8900-bdfe-11e8-8420-78ad9bb32f84.jpg
    :alt: norbert_wiener

Norbert is an implementation of multichannel Wiener filter, that is a very popular way of filtering multichannel audio for several applications, notably speech enhancement and source separation.

This repository assumes you have some way of estimating power-spectrograms for all the audio sources (non-negative) composing a mixture. It then builds the filter that is appropriate for extracting those signals from a mixture, by optimally exploiting multichannel information (like in stereo signals). This is done in an iterative procedure called _Expectation Maximization_, where filtering and re-estimation of the parameters are iterated.

Norbert implements mainly to core functions which are:

.. autosummary::

   norbert.softmask
   norbert.wiener

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

