import random
import time
from agents import agent
# StoryNode import is needed for unpickling the game file
from simulators.text_games.simulators.MySimulator import SavingJohnSimulator, StoryNode
import pickle

# This agent randomly searches the action space and remembers the best trace
# Best trace is a list of all game states and actions resulting in the largest cumulative reward
class RandomSearchAgent(agent.Agent):


    def __init__(self):
        random.seed(0)
        # we want to find the shortest best solution if possible
        self.stepCost = -0.001
        self.bestReward = -1000000
        self.bestTrace = []
        self.currentReward = 0
        self.currentTrace = []
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
        #print('total reward for last episode: ', self.currentReward)
        if self.currentTrace and self.currentReward > self.bestReward:
            self.bestReward = self.currentReward
            self.bestTrace = self.currentTrace
            print('new best reward : {0:10.3f}'.format(self.bestReward))
            print('new best actions: ', [x[1] for x in self.bestTrace])
        self.currentReward = 0;
        self.currentTrace = [];




def getSimulator():
    with open("simulators/text_games/simulators/savingjohn_wordId.pickle", "rb") as infile:
        dict_wordId = pickle.load(infile, encoding='utf-8')
    with open("simulators/text_games/simulators/savingjohn_actionId.pickle", "rb") as infile:
        dict_actionId = pickle.load(infile, encoding='utf-8')
    return SavingJohnSimulator(True), dict_wordId, dict_actionId, 4

def main():

    agent = RandomSearchAgent()
    startTime = time.time()
    mySimulator, dict_wordId, dict_actionId, maxNumActions = getSimulator()
    numEpisode = 0
    numStep = 0
    while numEpisode < 100000:
        (text, actions, reward) = mySimulator.Read()
        # print(text, actions, reward)
        if len(actions) == 0 or numStep > 250:
            agent.act(text, actions, reward)
            mySimulator.Restart()
            numEpisode += 1
            numStep = 0
            agent.reset()
        else:

            # playerInput = input()
            # playerInput = random.randint(0, len(actions) - 1)
            playerInput = agent.act(text, actions, reward)

            # print(actions[playerInput])

            mySimulator.Act(playerInput) # playerInput is index of selected actions
            numStep += 1

    endTime = time.time()
    print("Duration: " + str(endTime - startTime))


if __name__ == "__main__":
    main()
