class Game(object):
    name = ''
    author = ''
    path = ''
    # actions that should be entered when the game is started to accept various prompts, warning and settings
    startup_actions = []

    def __init__(self):
        raise NotImplementedError("Game is an abstract class.")


class CustomGame(Game):
    def __init__(self, name, path, author='', startup_actions=None):
        if startup_actions is None:
            startup_actions = []
        self.name = name
        self.path = path
        self.author = author
        self.startup_actions = startup_actions
