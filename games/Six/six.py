from games.game import Game
from interpreters.glulxe.glulxe import Glulxe



class Six(Game):
    name = 'Six'
    author = 'Wade Clarke'
    path = 'games/Six/Six.gblorb'
    interpreter = Glulxe
