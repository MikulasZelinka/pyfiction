import random
import time

# StoryNode import is needed for unpickling the game file:
import numpy as np
from pyfiction.agents import agent
from pyfiction.simulators.games.howlingdogs_simulator import HowlingDogsSimulator
from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator


# This agent randomly searches the action space and remembers the best trace and the best final reward for each state
# Best trace is a list of all game states and actions resulting in the largest cumulative reward
# Also finds the best possible cumulative reward for each visited state
from pyfiction.simulators.games.theredhair_simulator import TheRedHairSimulator


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

        self.trace.append([state, actions, reward, action])
        return index

    def reset(self):

        if not self.trace:
            return

        # if self.totalReward > self.bestReward:
        if True:
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

        self.totalReward = 0
        self.trace = []


        # MoD reward and ending checking:
        # check if all endings are annotated
        # ending = self.currentTrace[-1][0].split('THE END')[0]#.split('<html>')[-2]#.split('.')[-2]
        # found = False
        #
        # if """You spend your last few moments on Earth lying there, shot through the heart, by the image of Jon Bon Jovi.""" in ending:
        #     found = True
        # if """You may be locked away for some time.""" in ending:
        #     found = True
        # if """Eventually you're escorted into the back of a police car as Rachel looks on in horror.""" in ending:
        #     found = True
        # if """You can't help but smile.""" in ending:
        #     found = True
        # if """Fate can wait.""" in ending:
        #     found = True
        # if """you hear Bon Jovi say as the world fades around you.""" in ending:
        #     found = True
        # if """Hope you have a good life.""" in ending:
        #     found = True
        # if """As the screams you hear around you slowly fade and your vision begins to blur, you look at the words which ended your life.""" in ending:
        #     found = True
        # if """Sadly, you're so distracted with looking up the number that you don't notice the large truck speeding down the street.""" in ending:
        #     found = True
        # if """Stay the hell away from me!&quot; she blurts as she disappears into the crowd emerging from the bar.""" in ending:
        #     found = True
        # if """Congratulations!""" in ending:
        #     found = True
        # if """All these hiccups lead to one grand disaster.""" in ending:
        #     found = True
        # if """After all, it's your life. It's now or never. You ain't gonna live forever. You just want to live while you're alive.""" in ending:
        #     found = True
        # if """Rachel waves goodbye as you begin the long drive home. After a few minutes, you turn the radio on to break the silence.""" in ending:
        #     found = True
        #
        # if not found:
        #    print('ending with no reward found: ', ending)
        #
        # if ending not in self.endings:
        #    print('new ending: ', ending)
        #    self.endings.append(ending)
        #    print('endings count: ', len(self.endings))


def main():
    agent = RandomSearchAgent()
    start_time = time.time()
    # simulator = MachineOfDeathSimulator()
    # simulator = SavingJohnSimulator()
    # simulator = TheRedHairSimulator()
    simulator = HowlingDogsSimulator()
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

    print(agent.rewards)
    print(len(agent.states), ' states, ', len(agent.rewards), ' rewards')
    print(list(zip(agent.rewards, agent.states)))


if __name__ == "__main__":
    main()
