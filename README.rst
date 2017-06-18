pyfiction
---------

Python API for all kinds of text-based games.

Text-based games, more commonly known as interactive fiction (IF), come in various formats and use different interpreters that aren't often compatible with each other.

The goal of pyfiction is to wrap the functionality of different IF simulators and to provide a universal API for text games for research purposes.

Pyfiction also includes sample agents that can learn to play the supplied text games.

Status: Work in progress.

Requirements:

* Python 3.6 (older version might work but they aren't tested)
* `text-games <https://github.com/MikulasZelinka/text-games>`_ submodule for games 'Saving John' and 'Machine of Death'.
* `keras <https://github.com/fchollet/keras>`_, `tensorflow <https://github.com/tensorflow/tensorflow>`_ and their dependencies for launching the example agents.