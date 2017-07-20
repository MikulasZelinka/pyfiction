import random
import time

# StoryNode import is needed for unpickling the game file:
import numpy as np
from multiprocessing import Process
from pyfiction.agents import agent
from pyfiction.simulators.games.catsimulator2016_simulator import CatSimulator2016Simulator
from pyfiction.simulators.games.howlingdogs_simulator import HowlingDogsSimulator
from pyfiction.simulators.games.machineofdeath_simulator import MachineOfDeathSimulator
from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.games.starcourt_simulator import StarCourtSimulator
from pyfiction.simulators.games.theredhair_simulator import TheRedHairSimulator
from pyfiction.simulators.games.transit_simulator import TransitSimulator

from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode


class RandomSearchAgent(agent.Agent):
    """
    This agent randomly searches the action space and remembers:
     - the best trace and the best final cumulative reward for each state
         - best trace is a list of all game states and actions resulting in the largest cumulative reward
     - list of all endings
     - average reward
     - other useful statistic...

    This agent is very useful for determining various parameters of unknown games.
    """

    def __init__(self):
        random.seed(0)

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

        self.totalReward += reward

        action = None
        index = None

        if actions:
            index = random.randint(0, len(actions) - 1)
            action = actions[index]
            # print('choosing action:', action)

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
            # print('new ending: ', ending)
            self.endings.append(ending)
            # print('endings count: ', len(self.endings))

        # print()
        # print()

        self.totalReward = 0
        self.trace = []


def run(simulator, file_name, episodes, runs_per_episode, max_steps):
    agent = RandomSearchAgent()
    start_time = time.time()

    rewards = []
    words = 0
    descriptions = 0
    positives = 0

    for i in range(episodes):

        episode_rewards = []
        print('episode', i)

        for _ in range(runs_per_episode):

            steps = 0

            while True and steps < max_steps:

                (text, actions, reward) = simulator.read()
                # print(text, actions, reward)
                words += len(text.split())
                descriptions += 1

                player_input = agent.act(text, actions, reward)

                if player_input is None:
                    if agent.totalReward == 0:
                        agent.reset()
                        simulator.restart()
                        steps = 0
                        continue
                    else:
                        if agent.totalReward > 0:
                            positives += 1
                        episode_rewards.append(agent.totalReward)
                        agent.reset()
                        simulator.restart()
                        break

                else:
                    simulator.write(player_input)

                steps += 1

        rewards.append(episode_rewards)

    print('positive rewards', positives)

    with open(file_name, 'w') as f:
        # save rewards:
        # for episode_rewards in rewards:
        #     for reward in episode_rewards:
        #         f.write('{:.1f}'.format(reward) + ' ')
        #     f.write(',\n')
        f.write('{:.2f}'.format(words / descriptions))

    end_time = time.time()
    print("Duration: " + str(end_time - start_time))
    #
    # print()
    # print('Best rewards for all states:')
    # print(len(agent.states), ' states, ', len(agent.rewards), ' rewards')
    # print(agent.rewards)
    # print(list(zip(agent.rewards, agent.states)))
    # print()
    #
    # endings = agent.endings
    #
    # print('ENDINGS:', len(endings), ' ----------------------------- ')
    # for ending in endings:
    #     print(ending)
    #     print('*********************')


def runInParallel(*fns):
    proc = []
    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()


def main():
    MAsimulator = MachineOfDeathSimulator()
    # SAsimulator = SavingJohnSimulator()
    # THsimulator = TheRedHairSimulator()
    # simulator = HowlingDogsSimulator()
    # CAsimulator = CatSimulator2016Simulator()
    # TRsimulator = TransitSimulator()
    # STsimulator = StarCourtSimulator()

    episodes = 10000
    runs_per_episode = 1
    # max_steps = 500

    runInParallel(
        run(MAsimulator, '2random.txt', episodes, runs_per_episode, 500),
        # run(SAsimulator, '3random.txt', episodes, runs_per_episode, 100),
        # run(THsimulator, '5random.txt', episodes, runs_per_episode, 100),
        # run(CAsimulator, '1random.txt', episodes, runs_per_episode, 100),
        # run(TRsimulator, '6random.txt', episodes, runs_per_episode, 100),
        # run(STsimulator, '4random.txt', episodes, runs_per_episode, 500),
    )


if __name__ == "__main__":
    main()
