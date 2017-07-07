from pyfiction.games.SavingJohn.saving_john import SavingJohn
from pyfiction.simulators.simulator import Simulator
from pyfiction.simulators.text_games.simulators.MySimulator import SavingJohnSimulator as SJS


class SavingJohnSimulator(Simulator):
    """
    Simulator wrapper for the game Saving John.
    When importing this class, it is necessary to import StoryNode (!):
    from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode
    """

    # the maximum number of steps the agent should take before we interrupt him to break infinite cycles
    max_steps = 100

    # the recommended number of random game walkthroughs for vocabulary initialization
    # should ideally cover all possible states and used words
    initialization_iterations = 1024

    # if the game rewards are in e.g. [-30, 30], set the reward scale to 30 so that the result is in [-1, 1]
    reward_scale = 20

    def __startup_actions(self):
        pass

    def restart(self):
        self.simulator.Restart()

    def __init__(self, shuffle_actions=True):
        self.game = SavingJohn
        self.simulator = SJS(shuffle_actions)

    def read(self, **kwargs):
        return self.simulator.Read()

    def write(self, index):
        """
        Accepts an INDEX of an action, not its contents (i.e. not a string or a text description).
        """
        self.simulator.Act(index)
