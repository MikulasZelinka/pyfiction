import re

from pyfiction.simulators.simulator import Simulator
from pyfiction.simulators.text_games.simulators.MySimulator import MachineOfDeathSimulator as MODS


class MachineOfDeathSimulator(Simulator):
    """
    Simulator wrapper for the game Machine of Death.
    When importing this class, it is necessary to import StoryNode (!):
    from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode
    """

    def __startup_actions(self):
        pass

    def restart(self):
        self.simulator.Restart()

    def __init__(self, shuffle_actions=True):
        self.simulator = MODS(shuffle_actions)

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
