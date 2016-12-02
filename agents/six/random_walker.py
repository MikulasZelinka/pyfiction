import random
from subprocess import call

from agents import agent
from games.Six.six import Six
from simulator.simulator import Simulator


class RandomWalkerAgent(agent.Agent):
    def __init__(self):
        random.seed(0)
        self.actions = ['north\n', 'south\n', 'west\n', 'east\n']

    def act(self, observation):

        # handle special cases where a different output than a direction is needed
        if (observation[-1].endswith('(Y/N)\n')):
            action = 'N\n'

        else:
            action = self.actions[random.randint(0, 3)]

        return action


def main():
    # remove data from older test runs to always start the game from scratch
    call('rm -f *.glkdata', shell=True)

    simulator = Simulator(Six)
    simulator.start_game()
    simulator.startup_actions()

    agent = RandomWalkerAgent()

    observation = []

    while True:
        observation = simulator.read()
        # wait for input
        while not observation:
            observation = simulator.read()

        print(observation)
        action = agent.act(observation)
        print(action)
        simulator.write(action)


if __name__ == "__main__": main()
