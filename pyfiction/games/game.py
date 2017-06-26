class Game(object):
    name = ''
    author = ''
    url = ''
    path = ''
    # actions that should be entered when the game is started to accept various prompts, warning and settings
    startup_actions = []

    def __init__(self):
        raise NotImplementedError("Game is an abstract class.")


class CustomGame(Game):
    def __init__(self, name, author='', url='', path='', startup_actions=None):
        if startup_actions is None:
            startup_actions = []
        self.name = name
        self.author = author
        self.url = url
        self.path = path
        self.startup_actions = startup_actions
