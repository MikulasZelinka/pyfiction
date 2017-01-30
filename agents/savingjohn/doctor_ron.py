import random
import time
from agents import agent
# StoryNode import is needed for unpickling the game file
from simulators.text_games.simulators.MySimulator import SavingJohnSimulator, StoryNode
import pickle

# DoctoR RoN
class DoctorRonAgent(agent.Agent):
    def __init__(self):
        random.seed(0)

    def act(self, text, actions, reward):
        return random.randint(0, len(actions) - 1)


def getSimulator():
    with open("../../simulators/text_games/simulators/savingjohn_wordId.pickle", "rb") as infile:
        dict_wordId = pickle.load(infile, encoding='utf-8')
    with open("../../simulators/text_games/simulators/savingjohn_actionId.pickle", "rb") as infile:
        dict_actionId = pickle.load(infile, encoding='utf-8')
    return SavingJohnSimulator(True), dict_wordId, dict_actionId, 4

def main():

    agent = DoctorRonAgent()
    startTime = time.time()
    mySimulator, dict_wordId, dict_actionId, maxNumActions = getSimulator()
    numEpisode = 0
    numStep = 0
    while numEpisode < 10:
        (text, actions, reward) = mySimulator.Read()
        print(text, actions, reward)
        if len(actions) == 0 or numStep > 250:
            mySimulator.Restart()
            numEpisode += 1
            numStep = 0
        else:

            # playerInput = input()
            # playerInput = random.randint(0, len(actions) - 1)
            playerInput = agent.act(text, actions, reward)

            print(actions[playerInput])

            mySimulator.Act(playerInput) # playerInput is index of selected actions
            numStep += 1

    endTime = time.time()
    print("Duration: " + str(endTime - startTime))


if __name__ == "__main__":
    main()
