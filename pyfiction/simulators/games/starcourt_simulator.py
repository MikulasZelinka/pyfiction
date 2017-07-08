import random

import time
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException

from pyfiction.games.StarCourt.star_court import StarCourt
from pyfiction.simulators.html_simulator import HTMLSimulator
from pyfiction.simulators.simulator import UnknownEndingException


class StarCourtSimulator(HTMLSimulator):
    # the maximum number of steps the agent should take before we interrupt him to break infinite cycles
    max_steps = 500

    # the recommended number of random game walkthroughs for vocabulary initialization
    # should ideally cover all possible states and used words
    initialization_iterations = 4096

    # if the game rewards are in e.g. [-30, 30], set the reward scale to 30 so that the result is in [-1, 1]
    reward_scale = 30

    def __init__(self, shuffle=True):
        super().__init__(StarCourt, shuffle=shuffle)
        self.actions = None

    def restart(self):
        super().restart()

    def write(self, action_index):
        action = self.actions[action_index][1]
        action.click()

    def read(self, tries=0, max_tries=10):

        try:
            # text is always in the the tw-story html tag:
            passage = self.driver.find_element_by_class_name("passage")
            text = passage.text

            # self.actions = [(action.text, action) for action in passage.find_elements_by_class_name("internalLink")]
            self.actions = [(action.text, action) for action in passage.find_elements_by_tag_name("a")]

            reward = -0.1

            # reward for using favors to upgrade the job/pet/house:
            if text.startswith('You get a job as a'):
                reward += 5

            # detect endings
            if len(self.actions) == 2 and self.actions[0][0] == "Take survey." and self.actions[1][0] == "Start over.":
                self.actions = []

            if not self.actions:
                ending = text

                # death but no sentence
                if ending.startswith(
                    'Here on the astral plane, your psychic bodies are as physical and real as your material body is in the physical world.'):
                    reward = -20
                # death sentence by trial (literally)
                elif ending.startswith(
                    'Nah. You die as poison consumes your body. And because you failed trial by poison, you\'re declared guilty!'):
                    reward = -30
                # happily ever after
                elif ending.startswith('You\'re all out of favors! I guess working as a'):
                    reward = 15
                # death sentence by trial (literally)
                elif ending.startswith(
                    'The only thing Pride finds more beautiful than itself is the destruction of those less beautiful than it!'):
                    reward = -30
                # death sentence by trial (literally)
                elif ending.startswith('Immediately upon starting the battle, the titanic creature falls asleep!'):
                    reward = -30
                # death sentence by trial (literally)
                elif ending.startswith('You are torn limb from limb by the many-limbed creature!'):
                    reward = -30
                # "blasting you right in the core"
                elif ending.startswith('You remember you training at Psi City and concentrate'):
                    reward = -30
                # innocent
                elif ending.startswith('And so you do, spacer, so you do.'):
                    reward = 15
                # speaks for itself
                elif ending.startswith("BLAMMO!!\n\nYou're dead! And what's worse, you're guilty!"):
                    reward = -30

                # sentenced but alive, determine the reward based on length of the sentence: (0 < length < 2000) years mapped to (-20 < reward < 0)
                elif 'The Judge bangs their laser gavel a final time. "Robailiff, you may take the prisoner away."' in ending:
                    words = ending.split()
                    years = int(words[words.index('years') - 1])
                    reward = -years / 100

                # one of the many endings of guilt and death
                elif "You're dead! I guess that means you're guilty!" in ending:
                    reward = -30
                # dead but not guilty
                elif 'You are neither guilty nor innocent, as law has been dethroned in the universe. However, you have died in a fire.' in ending:
                    reward = -20
                # escaped
                elif "You let Star Court evaporate like a bad memory. You're on the other side of the sector by the time they notice you're missing." in ending:
                    reward = 10
                # burned
                elif 'How does Star Court generate this much trash, you think as you burn.' in ending:
                    reward = -20
                # death
                elif 'You got smoked by a crime ghost.' in ending:
                    reward = -20
                # death
                elif "Congratulations, you're innocent! You're also dead." in ending:
                    reward = -20
                # dead and guilty
                elif "The knife hits you right between the eyes. You are killed immediately, and, what's worse, you're found guilty by the court!" in ending:
                    reward = -30
                # dead and guilty
                elif "means you're guilty!" in ending:
                    reward = -30

                else:
                    raise UnknownEndingException('Unknown ending text, cannot assign reward: ', ending)

            elif self.shuffle:
                random.shuffle(self.actions)

        except (UnknownEndingException, NoSuchElementException, StaleElementReferenceException) as e:

            if tries == 0:
                print('WARNING, simulator exception:', e)

            if tries < max_tries:
                print('Trying to read again after a short wait, try', tries + 1, 'out of', max_tries)
                time.sleep(0.1)
                return self.read(tries=tries + 1)
            else:
                raise e

        return text, [action[0] for action in self.actions], reward

    def close(self):
        self.driver.close()


if __name__ == '__main__':
    simulator = StarCourtSimulator()
    starts = 0
    ends = 0

    for i in range(100):

        total_reward = 0

        while True:

            state, actions, reward = simulator.read()
            total_reward += reward

            if state.startswith('‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌The laser gavel pounds.'):
                starts += 1

            # print(state)
            # print('actions: ', actions)
            # print(reward)
            # print('-----------------------------')

            if not actions:
                print(state)
                print('last', reward, 'total', total_reward)
                print('--------------------')
                ends += 1
                break

            action = random.randint(0, len(actions) - 1)
            simulator.write(action)

        simulator.restart()

    simulator.driver.close()
    print('starts', starts, 'ends', ends)
