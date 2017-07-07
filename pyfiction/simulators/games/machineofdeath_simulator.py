import re

from pyfiction.games.MachineOfDeath.machine_of_death import MachineOfDeath
from pyfiction.simulators.simulator import Simulator
from pyfiction.simulators.text_games.simulators.MySimulator import MachineOfDeathSimulator as MODS


class MachineOfDeathSimulator(Simulator):
    """
    Simulator wrapper for the game Machine of Death.
    When importing this class, it is necessary to import StoryNode (!):
    from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode
    """

    # the maximum number of steps the agent should take before we interrupt him to break infinite cycles
    max_steps = 500

    # the recommended number of random game walkthroughs for vocabulary initialization
    # should ideally cover all possible states and used words
    initialization_iterations = 8192

    # if the game rewards are in e.g. [-30, 30], set the reward scale to 30 so that the result is in [-1, 1]
    reward_scale = 30

    def __startup_actions(self):
        pass

    def restart(self):
        self.simulator.Restart()

    def __init__(self, shuffle_actions=True, paraphrase_actions=False):
        self.game = MachineOfDeath
        self.simulator = MODS(shuffle_actions, doParaphrase=paraphrase_actions)

    def read(self, **kwargs):
        state, actions, reward = self.simulator.Read()
        # the original simulator does not remove some HTML tags - remove them all:
        state = re.sub('<[^>]*>', '', state)
        return state, actions, reward

    def write(self, index):
        """
        Accepts an INDEX of an action, not its contents (i.e. not a string or a text description).
        """
        self.simulator.Act(index)
