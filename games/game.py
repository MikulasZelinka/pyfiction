class Game(object):
    name = ''
    author = ''
    path = ''
    interpreter = None
    # actions that should be entered when the game is started to accept various prompts, warning and settings
    startup_actions = []

    def __init__(self):
        raise NotImplementedError("Game is an abstract class.")
