from games.game import Game
from interpreters.glulxe.glulxe import Glulxe


class Six(Game):
    name = 'Six'
    author = 'Wade Clarke'
    path = 'games/Six/Six.gblorb'
    interpreter = Glulxe

    # issue extra SPACE commands just to be sure
    startup_actions = [' \n', ' \n', ' \n', ' \n', '1\n', ' \n']
