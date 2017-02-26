import random
import time
from pyfiction.agents import agent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode


class DoctorRonAgent(agent.Agent):
    """
    DRRN agent
    """

    def __init__(self):
        random.seed(0)

    def act(self, text, actions, reward):
        return random.randint(0, len(actions) - 1)


def main():
    agent = DoctorRonAgent()
    startTime = time.time()
    mySimulator = SavingJohnSimulator()
    numEpisode = 0
    numStep = 0
    while numEpisode < 10:
        (text, actions, reward) = mySimulator.read()
        print(text, actions, reward)
        if len(actions) == 0 or numStep > 250:
            mySimulator.restart()
            numEpisode += 1
            numStep = 0
        else:

            # playerInput = input()
            # playerInput = random.randint(0, len(actions) - 1)
            playerInput = agent.act(text, actions, reward)

            print(actions[playerInput])

            mySimulator.write(playerInput)  # playerInput is index of selected actions
            numStep += 1

    endTime = time.time()
    print("Duration: " + str(endTime - startTime))


if __name__ == "__main__":
    main()
