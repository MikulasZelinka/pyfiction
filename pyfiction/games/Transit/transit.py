import os

from pyfiction.games.game import Game


class Transit(Game):

    def __init__(self):
        pass

    name = 'Transit'
    author = 'Shaye'
    url = 'http://ifdb.tads.org/viewgame?id=stkrrbel8m21b37q'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')

    startup_actions = []
