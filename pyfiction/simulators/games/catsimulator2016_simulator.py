import random

import time

from selenium.common.exceptions import NoSuchElementException

from pyfiction.games.CatSimulator2016.cat_simulator_2016 import CatSimulator2016
from pyfiction.simulators.html_simulator import HTMLSimulator
from pyfiction.simulators.simulator import UnknownEndingException


class CatSimulator2016Simulator(HTMLSimulator):
    # the maximum number of steps the agent should take before we interrupt him to break infinite cycles
    max_steps = 100

    # the recommended number of random game walkthroughs for vocabulary initialization
    # should ideally cover all possible states and used words
    initialization_iterations = 256

    # if the game rewards are in e.g. [-30, 30], set the reward scale to 30 so that the result is in [-1, 1]
    reward_scale = 20

    def __init__(self, shuffle_actions=True):
        super().__init__(CatSimulator2016, shuffle_actions=shuffle_actions)
        self.actions = None

    def restart(self):
        super().restart()

    def write(self, action_index):
        action = self.actions[action_index][1]
        action.click()

    def read(self, tries=0, max_tries=10):

        try:

            # text is always in the the tw-story html tag:
            text = self.driver.find_element_by_tag_name("tw-story").text

            self.actions = []

            try:
                # actions are always of one of the two class sets below:
                self.actions += [(action.text, action) for action in
                                 self.driver.find_elements_by_tag_name("tw-link")]

                self.actions += [(action.text, action) for action in
                                 self.driver.find_elements_by_css_selector(
                                     "tw-enchantment[class='link enchantment-link']")]
            except:
                pass

            reward = -0.1

            # detect good endings (bad endings have no actions)
            if len(self.actions) == 1 and self.actions[0][0] == "spoilers":
                self.actions = []

            if not self.actions:
                ending = text.lower()

                # remove the first line of the ending (only contains a back arrow):
                ending = ending[(ending.index('\n') + 1):]

                # slept on the stool but no food
                if ending.startswith('this was a good idea'):
                    reward = 0
                # fell down and slept briefly - no food
                elif ending.startswith('as good a place as any'):
                    reward = -20
                # killed the bird
                elif ending.startswith('mine!'):
                    reward = 10
                # fell into the sink - no food and no sleep
                elif ending.startswith('catlike reflexes'):
                    reward = -20
                # failed to hunt the bird - no food and no sleep
                elif ending.startswith('finish this'):
                    reward = -20
                # befriended the bird - food and sleep
                elif ending.startswith('friendship'):
                    reward = 20
                # no food, slept on the counter
                elif ending.startswith('not this time, water'):
                    reward = 10
                # slept outside, no food
                elif ending.startswith('serendipity'):
                    reward = 10
                else:
                    raise UnknownEndingException('Unknown ending text, cannot assign reward: ', ending)

            elif self.shuffle_actions:
                random.shuffle(self.actions)

        except (UnknownEndingException, NoSuchElementException) as e:

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
    simulator = CatSimulator2016Simulator()

    for i in range(16):

        while True:

            try:
                state, actions, reward = simulator.read()
            except:
                pass

            print(state)
            for action in actions:
                print(action)
            print('actions: ', actions)
            print(reward)
            print('-----------------------------')

            if not actions:
                break

            action = random.randint(0, len(actions) - 1)
            simulator.write(action)

        simulator.restart()

    simulator.driver.close()
