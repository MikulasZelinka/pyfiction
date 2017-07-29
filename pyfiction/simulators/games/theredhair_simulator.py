import random

import time
from selenium.common.exceptions import NoSuchElementException

from pyfiction.games.TheRedHair.the_red_hair import TheRedHair
from pyfiction.simulators.html_simulator import HTMLSimulator
from pyfiction.simulators.simulator import UnknownEndingException


class TheRedHairSimulator(HTMLSimulator):
    # the maximum number of steps the agent should take before we interrupt him to break infinite cycles
    max_steps = 100

    # the recommended number of random game walkthroughs for vocabulary initialization
    # should ideally cover all possible states and used words
    initialization_iterations = 64

    # if the game rewards are in e.g. [-30, 30], set the reward scale to 30 so that the result is in [-1, 1]
    reward_scale = 20

    def __init__(self, shuffle_actions=True):
        super(TheRedHairSimulator, self).__init__(TheRedHair, shuffle_actions=shuffle_actions)

    def restart(self):
        # super().restart()
        # must restart in loop in case the button is not available yet
        while True:
            restarted = True
            try:
                self.driver.find_element_by_id("restart").click()
            except:
                restarted = False
            if restarted:
                break

    def write(self, action_index):
        action = self.actions[action_index][1]
        # action.click()
        # click in javascript instead (in chrome, clicks don't work here for some strange reason - hidden by ps or divs)
        self.driver.execute_script("arguments[0].click();", action)

    def read(self, tries=0, max_tries=10):

        try:
            # text is always in the last div:
            text = (self.driver.find_elements_by_css_selector("div")[-1]).text

            self.actions = []

            # don't fail if there are no actions of the second class
            try:
                # actions are always of one of the two class sets below:
                self.actions += [(action.text, action) for action in
                                 self.driver.find_elements_by_css_selector("a[class='squiffy-link link-section']")]
                self.actions += [(action.text, action) for action in
                                 self.driver.find_elements_by_css_selector("a[class='squiffy-link link-passage']")]
            except:
                pass

            reward = -0.1

            if not self.actions:
                ending = text.lower()
                if ending.startswith('you lose'):
                    reward = -10
                elif ending.endswith('all there is left is a red hair') or ending.endswith(
                    'it was the clown statue missing'):
                    reward = -20
                elif text.lower().startswith(
                    'you stay in the bedroom and eventually the parents come back and thank you'):
                    reward = 20
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
    simulator = TheRedHairSimulator()

    for i in range(16):

        while True:
            state, actions, reward = simulator.read()

            print(state)
            for action in actions:
                print(action)
            print(reward)

            if not actions:
                break

            action = random.randint(0, len(actions) - 1)
            simulator.write(action)

        simulator.restart()

    simulator.driver.close()
