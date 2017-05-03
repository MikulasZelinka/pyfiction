import random
import time
from pyfiction.agents import agent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode


# This agent randomly searches the action space and remembers the best trace
# Best trace is a list of all game states and actions resulting in the largest cumulative reward
class RandomSearchAgent(agent.Agent):
    def __init__(self):
        random.seed(0)
        # we want to find the shortest best solution if possible
        self.stepCost = -0.01
        self.bestReward = -1000000
        self.bestTrace = []
        self.currentReward = 0
        self.currentTrace = []
        self.totalReward = 0
        self.count = 0
        self.reset()

    def act(self, text, actions, reward):
        self.currentReward += reward + self.stepCost
        actionText = ''
        if actions:
            actionID = random.randint(0, len(actions) - 1)
            actionText = actions[actionID]
            self.currentTrace.append([text, actionText])
            return actionID

    def reset(self):
        # print('total reward for last episode: ', self.currentReward)
        self.totalReward += self.currentReward
        self.count += 1
        if self.currentTrace and self.currentReward > self.bestReward:
            self.bestReward = self.currentReward
            self.bestTrace = self.currentTrace
            print('new best reward : {0:10.3f}'.format(self.bestReward))
            print('new best actions: ', [x[1] for x in self.bestTrace])
        self.currentReward = 0
        self.currentTrace = []


def main():
    agent = RandomSearchAgent()
    start_time = time.time()
    my_simulator = SavingJohnSimulator()
    num_episode = 0
    num_step = 0
    while num_episode < 100000:
        (text, actions, reward) = my_simulator.read()
        # print(text, actions, reward)
        if len(actions) == 0 or num_step > 250:
            agent.act(text, actions, reward)
            my_simulator.restart()
            num_episode += 1
            num_step = 0
            agent.reset()
        else:

            # playerInput = input()
            # playerInput = random.randint(0, len(actions) - 1)
            playerInput = agent.act(text, actions, reward)

            # print(actions[playerInput])

            my_simulator.write(playerInput)  # playerInput is index of selected actions
            num_step += 1

    end_time = time.time()
    print("Duration: ", (end_time - start_time), " seconds")
    print("Average reward: ", agent.totalReward / agent.count)


if __name__ == "__main__":
    main()
