import os

from pyfiction import PYFICTION_PATH
from pyfiction.games.game import Game


class Six(Game):

    def __init__(self):
        pass

    name = 'Six'
    author = 'Wade Clarke'
    path = os.path.join(PYFICTION_PATH, 'games/Six/Six.gblorb')

    # issue extra SPACE commands just to be sure
    startup_actions = [' \n', ' \n', ' \n', ' \n', '1\n', ' \n']
