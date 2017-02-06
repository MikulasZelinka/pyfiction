class Simulator:
    def __init__(self):
        raise NotImplementedError("Game is an abstract class.")

    def start_game(self):
        raise NotImplementedError("Game is an abstract class.")

    def startup_actions(self):
        raise NotImplementedError("Game is an abstract class.")

    def write(self, text):
        raise NotImplementedError("Game is an abstract class.")

    def read(self, timeout=0.01):
        raise NotImplementedError("Game is an abstract class.")
