=========
pyfiction
=========

Python API for all kinds of text-based games.


Introduction
------------
Text-based games, more commonly known as interactive fiction (IF), come in various formats and use different interpreters that aren't often compatible with each other.

The goal of pyfiction is to wrap the functionality of different IF simulators and to provide a universal API for text games for research purposes.

Pyfiction also includes sample agents that can learn to play the supplied text games.

Requirements
------------

* Python 3.6 (older version might work but they aren't tested, Python 3.5 tested briefly)
* `text-games <https://github.com/MikulasZelinka/text-games>`_ submodule for games 'Saving John' and 'Machine of Death'
* `keras <https://github.com/fchollet/keras>`_, `tensorflow <https://github.com/tensorflow/tensorflow>`_ and their dependencies for launching the example agents
* selenium and `chromedriver <https://sites.google.com/a/chromium.org/chromedriver/>`_ for running the HTML-based games
* h5py, pydot (optional)

Installation
------------

Clone the repository and install the library using: ::

  git clone --recursive https://github.com/MikulasZelinka/pyfiction
  cd pyfiction
  pip install [--user] -e .

Examples
--------

To run any example (from the game list below or from this list), simply run: ::

  python file.py


on these examples:

* `Agent playing multiple games <pyfiction/examples/generalisation/generalisation.py>`_,
* `Interactive testing <pyfiction/examples/generalisation/interactive_test.py>`_ of Q-values for states and actions of an agent trained on all six games below,

or on one of the supported games below.

Currently supported games
-------------------------
The following links lead to examples of training experiments for the games:

* `Saving John <pyfiction/examples/savingjohn/lstm_online.py>`_
* `Machine of Death <pyfiction/examples/machineofdeath/lstm_online.py>`_
* `Cat Simulator 2016 <pyfiction/examples/catsimulator2016/lstm_online.py>`_ [HTML-based]
* `Star Court <pyfiction/examples/starcourt/lstm_online.py>`_ [HTML-based]
* `The Red Hair <pyfiction/examples/theredhair/lstm_online.py>`_ [HTML-based]
* `Transit <pyfiction/examples/transit/lstm_online.py>`_ [HTML-based]

Adding new games
~~~~~~~~~~~~~~~~

TODO.


Status
------
Version 0.1.0, Alpha


References
----------

* `Language Understanding for Text-based Games Using Deep Reinforcement Learning <https://arxiv.org/abs/1506.08941>`_
* `Deep Reinforcement Learning with a Natural Language Action Space <https://arxiv.org/abs/1511.04636>`_
* `Using reinforcement learning to play text-based games <http://www.ms.mff.cuni.cz/~zelinkm/text-games/>`_ (available in 09/2017)