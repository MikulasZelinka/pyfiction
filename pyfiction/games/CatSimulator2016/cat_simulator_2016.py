import os

from pyfiction.games.game import Game


class CatSimulator2016(Game):

    def __init__(self):
        pass

    name = 'Cat Simulator 2016'
    author = 'helado de brownie'
    url = 'http://ifdb.tads.org/viewgame?id=79f1ic623cvxtpio'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')

    startup_actions = []
