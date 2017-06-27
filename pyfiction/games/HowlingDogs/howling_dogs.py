import os

from pyfiction.games.game import Game


class HowlingDogs(Game):

    def __init__(self):
        pass

    name = 'howling dogs'
    author = 'Porpentine'
    url = 'ifdb.tads.org/viewgame?id=mxj7xp4nffia9rbj'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')

    # startup_actions = ['still', 'howling dogs']
    # corresponding action indices:
    startup_actions = [0, 0]
