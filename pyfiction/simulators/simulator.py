class Simulator:
    def __init__(self):
        raise NotImplementedError("Game is an abstract class.")

    def restart(self):
        raise NotImplementedError("Game is an abstract class.")

    def __startup_actions(self):
        raise NotImplementedError("Game is an abstract class.")

    def write(self, text):
        raise NotImplementedError("Game is an abstract class.")

    def read(self, timeout=0.01):
        raise NotImplementedError("Game is an abstract class.")
