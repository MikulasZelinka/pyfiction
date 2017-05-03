import random
import time
from pyfiction.agents import agent
# StoryNode import is needed for unpickling the game file
from pyfiction.simulators.text_games.simulators.MySimulator import MachineOfDeathSimulator, StoryNode
import pickle


# This agent randomly searches the action space and remembers the best trace
# Best trace is a list of all game states and actions resulting in the largest cumulative reward
class RandomSearchAgent(agent.Agent):
    def __init__(self):
        random.seed(0)
        # we want to find the shortest best solution if possible
        self.stepCost = 0
        self.bestReward = -1000000
        self.bestTrace = []

        # reward and trace for one episode
        self.totalReward = 0
        self.trace = []

        # current text and current chosen action descriptions
        self.text = ''
        self.action = ''

        self.endings = []
        # experience is a set of all traces, where a trace is a set of tuples: [state, action, reward]
        self.experience = []
        self.reset()

    def act(self, text, actions, reward):

        # store: s, a, r, s', a'
        self.trace.append([self.text, self.action, reward, text, actions])

        self.text = text

        self.totalReward += reward + self.stepCost

        if actions:
            actionID = random.randint(0, len(actions) - 1)
            self.action = actions[actionID]
            return actionID

    def reset(self):
        # print('total reward for last episode: ', self.currentReward)

        if self.trace:

            if self.totalReward > self.bestReward:
                self.bestReward = self.totalReward
                self.bestTrace = self.trace
                print('new best reward : {0:10.3f}'.format(self.bestReward))
                print('new best actions: ', [x[1] for x in self.bestTrace])
                print('last state: ', self.trace[-1][0])

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

            self.experience.append(self.trace)

        self.totalReward = 0
        self.trace = []


def getSimulator():
    with open("../../simulators/text_games/simulators/machineofdeath_wordId.pickle", "rb") as infile:
        dict_wordId = pickle.load(infile, encoding='utf-8')
    with open("../../simulators/text_games/simulators/machineofdeath_actionId.pickle", "rb") as infile:
        dict_actionId = pickle.load(infile, encoding='utf-8')
    return MachineOfDeathSimulator(doShuffle=True, doParaphrase=False), dict_wordId, dict_actionId, 4


def main():
    agent = RandomSearchAgent()
    startTime = time.time()
    mySimulator, dict_wordId, dict_actionId, maxNumActions = getSimulator()
    numEpisode = 0
    totalReward = 0
    steps = 10000
    while numEpisode < steps:
        (text, actions, reward) = mySimulator.Read()
        # print(text, actions, reward)

        totalReward += reward

        # the game has ended
        if len(actions) == 0:

            agent.act(text, actions, reward)
            mySimulator.Restart()
            numEpisode += 1
            agent.reset()

        # choose an action
        else:

            # playerInput = input()
            # playerInput = random.randint(0, len(actions) - 1)
            playerInput = agent.act(text, actions, reward)

            # print(actions[playerInput])

            mySimulator.Act(playerInput)  # playerInput is index of selected actions

    endTime = time.time()
    print("Average reward: ", totalReward / steps)
    print("Duration: " + str(endTime - startTime))


if __name__ == "__main__":
    main()
