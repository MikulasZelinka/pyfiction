import random
import time

# StoryNode import is needed for unpickling the game file:
import numpy as np
from pyfiction.agents import agent
from pyfiction.simulators.games.catsimulator2016_simulator import CatSimulator2016Simulator
from pyfiction.simulators.games.howlingdogs_simulator import HowlingDogsSimulator
from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.games.theredhair_simulator import TheRedHairSimulator

# This agent randomly searches the action space and remembers:
#  - the best trace and the best final cumulative reward for each state
#      - best trace is a list of all game states and actions resulting in the largest cumulative reward
#  - list of all endings
#
# The random search is very useful when testing a new game to find all its endings and find out what states have what
# possible rewards.



class RandomSearchAgent(agent.Agent):
    def __init__(self):
        random.seed(0)

        # we want to find the shortest best solution if possible - use stepCost
        self.stepCost = -0.1
        self.bestReward = -np.math.inf
        self.bestTrace = []

        # reward and trace for one episode
        self.totalReward = 0
        self.trace = []

        self.endings = []
        # experience is a set of all traces, where a trace is a set of tuples: [state, action, reward]
        self.experience = []

        self.states = []
        self.rewards = []

        self.reset()

    def act(self, state, actions, reward):

        if reward == 0:
            reward = self.stepCost

        self.totalReward += reward

        action = None
        index = None

        if actions:
            index = random.randint(0, len(actions) - 1)
            action = actions[index]
            print('choosing action:', action)

        self.trace.append([state, actions, reward, action])
        return index

    def reset(self):

        if not self.trace:
            return

        if self.totalReward > self.bestReward:
            self.bestReward = self.totalReward
            self.bestTrace = self.trace
            print('new best reward : {0:10.3f}'.format(self.bestReward))
            print('new best actions: ', [x[3] for x in self.bestTrace])
            print('last state: ', self.trace[-1][0])
            print('----')
            print(self.trace[-10:])
            print('----')

        for state, _, _, _ in self.trace:
            if state in self.states:
                index = self.states.index(state)
                self.rewards[index] = max(self.totalReward, self.rewards[index])
            else:
                self.states.append(state)
                self.rewards.append(self.totalReward)

                # self.experience.append(self.trace)

        ending = self.trace[-1][0]

        if ending not in self.endings:
           print('new ending: ', ending)
           self.endings.append(ending)
           print('endings count: ', len(self.endings))

        print()
        print()

        self.totalReward = 0
        self.trace = []




def main():
    agent = RandomSearchAgent()
    start_time = time.time()
    # simulator = MachineOfDeathSimulator()
    # simulator = SavingJohnSimulator()
    # simulator = TheRedHairSimulator()
    # simulator = HowlingDogsSimulator()
    simulator = CatSimulator2016Simulator()
    num_episode = 0
    episodes = 2 ** 6
    while num_episode < episodes:

        (text, actions, reward) = simulator.read()
        # print(text, actions, reward)

        player_input = agent.act(text, actions, reward)

        if player_input is None:
            agent.reset()
            simulator.restart()
            num_episode += 1

        else:
            simulator.write(player_input)

    end_time = time.time()
    print("Duration: " + str(end_time - start_time))

    print()
    print('Best rewards for all states:')
    print(len(agent.states), ' states, ', len(agent.rewards), ' rewards')
    print(agent.rewards)
    print(list(zip(agent.rewards, agent.states)))
    print()

    print('ENDINGS:', len(agent.endings), ' ----------------------------- ')
    for ending in agent.endings:
        print(ending)
        print('*********************')


if __name__ == "__main__":
    main()
