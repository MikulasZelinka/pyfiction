class Simulator:

    # the maximum number of steps the agent should take before we interrupt him to break infinite cycles
    max_steps = 500

    # the recommended number of random game walkthroughs for vocabulary initialization
    # should ideally cover all possible states and used words
    initialization_iterations = 1024

    # if the game rewards are in e.g. [-30, 30], set the reward scale to 30 so that the result is in [-1, 1]
    reward_scale = 1

    # reference to the underlying game class
    game = None

    def __init__(self):
        raise NotImplementedError("Game is an abstract class.")

    def restart(self):
        raise NotImplementedError("Game is an abstract class.")

    def startup_actions(self):
        raise NotImplementedError("Game is an abstract class.")

    def write(self, text):
        raise NotImplementedError("Game is an abstract class.")

    def read(self, timeout=0.01):
        raise NotImplementedError("Game is an abstract class.")


class UnknownEndingException(Exception):
    """Reached a state without any actions but cannot assign a reward, unknown state"""
