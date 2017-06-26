import os

from pyfiction.games.game import Game


class TheRedHair(Game):

    def __init__(self):
        pass

    name = 'The Red Hair'
    author = 'Baboon Ben'
    url = 'http://textadventures.co.uk/games/view/r0fika63aksao_qkq8n3lq/the-red-hair'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')

    startup_actions = []
