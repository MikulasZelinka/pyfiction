import os

from pyfiction.games.game import Game


class StarCourt(Game):

    def __init__(self):
        pass

    name = 'Star Court'
    author = 'Anna Anthropy'
    url = 'http://ifdb.tads.org/viewgame?id=u1v4q16f7gujdb2g'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')

    startup_actions = []
