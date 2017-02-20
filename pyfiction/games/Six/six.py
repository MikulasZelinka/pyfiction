import os

from pyfiction.games.game import Game


class Six(Game):

    def __init__(self):
        pass

    name = 'Six'
    author = 'Wade Clarke'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Six.gblorb')

    # issue extra SPACE commands just to be sure
    startup_actions = [' \n', ' \n', ' \n', ' \n', '1\n', ' \n']
