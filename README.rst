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
* h5py for saving and loading models
* pydot for visualising model graphs

Installation
------------

To install the latest published version, simply run: ::

  pip install pyfiction


To install the bleeding-edge version from this repository, run: ::

  git clone --recursive https://github.com/MikulasZelinka/pyfiction
  cd pyfiction
  pip install -e .

Interface
---------

*pyfiction* provides a simple agent-environment interface for text-game environments.


.. figure:: http://web.stanford.edu/group/pdplab/pdphandbookV3/suttonbarto_rl.png
   :align: center
   :alt: agent-environment interface

   Simple RL interface [4]. In *pyfiction*, the agent calls the corresponding functions of the game environment:

   * ``state, actions, rewards = game.read()`` (left branch),
   * ``game.write(action)`` (right branch).

Examples
--------

To run any example (from the game list below or from this list), simply run: ::

  python file.py

on these files:

* `Interactively play any of the supported games <pyfiction/examples/interactive.py>`_,
* `An SSAQN (siamese state-action q-network) agent [3] learning to play multiple games at once <pyfiction/examples/generalisation/generalisation.py>`_,
* `Interactive testing <pyfiction/examples/generalisation/interactive_test.py>`_ of state-action Q-values of the above agent trained on the supported games,

or on one of the supported games below.

Currently supported games
-------------------------
The following links lead to training experiments of the SSAQN agent on the supported games:

* `Saving John <pyfiction/examples/savingjohn/lstm_online.py>`_
* `Machine of Death <pyfiction/examples/machineofdeath/lstm_online.py>`_
* `Cat Simulator 2016 <pyfiction/examples/catsimulator2016/lstm_online.py>`_ [HTML-based]
* `Star Court <pyfiction/examples/starcourt/lstm_online.py>`_ [HTML-based]
* `The Red Hair <pyfiction/examples/theredhair/lstm_online.py>`_ [HTML-based]
* `Transit <pyfiction/examples/transit/lstm_online.py>`_ [HTML-based]

Additionally, the plan is to integrate the individual games to the `OpenAI Gym <https://github.com/openai/gym>`_, see
the `pull request <https://github.com/openai/gym/pull/671>`_ and the `text_games branch <https://github.com/MikulasZelinka/gym/tree/text_games>`_ for details.

Adding new games
~~~~~~~~~~~~~~~~

TODO.



References
----------

[1] `Language Understanding for Text-based Games Using Deep Reinforcement Learning <https://arxiv.org/abs/1506.08941>`_

[2] `Deep Reinforcement Learning with a Natural Language Action Space <https://arxiv.org/abs/1511.04636>`_

[3] `Using reinforcement learning to learn how to play text-based games <http://www.ms.mff.cuni.cz/~zelinkm/text-games/thesis.pdf>`_ (Master thesis, available in 09/2017)

[4] `Reinforcement Learning: An Introduction <http://incompleteideas.net/sutton/book/the-book-2nd.html>`_


Status
------
Version 0.1.2, Alpha