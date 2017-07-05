import random

import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

from pyfiction.games.Transit.transit import Transit
from pyfiction.simulators.html_simulator import HTMLSimulator


class TransitSimulator(HTMLSimulator):
    def __init__(self, shuffle=True):
        super().__init__(Transit, shuffle=shuffle)

        self.actions = []

        # the game loops infinitely if a same action is selected twice, we remove these actions to avoid infinite cycles
        self.actions_history = []

    def restart(self):
        super().restart()
        self.actions_history = []

    def write(self, action_index):
        action = self.actions[action_index][1]
        action_text = self.actions[action_index][0]
        # print('choosing', action_text)

        # action = self.driver.find_elements_by_link_text(action_text)[-1]

        self.actions_history.append(action_text)
        action.click()

    def read(self):

        # text is always in the last passage div:
        last_state = self.driver.find_elements_by_class_name("passage")[-1]
        text = last_state.text
        self.actions = []

        self.actions += [(action.text, action) for action in last_state.find_elements_by_tag_name("a") if
                         action.text]

        had_actions = len(self.actions) > 0

        # filter the already used actions (workaround for the game bug that causes infinite loops)
        self.actions = [action for action in self.actions if action[0] not in self.actions_history]

        reward = -0.1

        if had_actions and not self.actions:
            print('WARNING: Transit simulator removed all actions because of a game bug! Ending with a 0 reward.')
            return '', [], 0


        if not self.actions:
            ending = text.lower()

            # tackled a man, found the friend
            if ending.startswith('if anyone can help you'):
                reward = 10
            # death by poisoning
            elif ending.startswith('you buy one more can'):
                reward = -20
            # tackled the correct man
            elif ending.startswith('even though it was just in-passing'):
                reward = 20
            # shot by the security
            elif ending.startswith('you make swift use of'):
                reward = -20
            # jail
            elif ending.startswith('the guards know'):
                reward = -10
            # jail
            elif ending.startswith('as you predicted'):
                reward = -10
            # death in a foreign country
            elif ending.endswith('you close your eyes and submit to death.'):
                reward = -20
            # jail in a foreign country
            elif ending.startswith('you\'re in a country'):
                reward = -10
            # escaped to the plane with the help of energy drinks
            elif ending.startswith('through the haze of the drinks'):
                reward = 10
            # ended in a jail with the help of energy drinks
            elif ending.startswith('while the last parts of your mind untouched'):
                reward = -10
            else:
                raise Exception('Game ended and no actions left but an unknown ending reached, cannot assign reward: ',
                                ending)
                # print('Game ended and no actions left but an unknown ending reached, cannot assign reward: ',
                #       ending)
                # pass



        elif self.shuffle:
            random.shuffle(self.actions)

        return text, [action[0] for action in self.actions], reward

    def close(self):
        self.driver.close()


if __name__ == '__main__':
    simulator = TransitSimulator()

    for i in range(16):

        while True:

            # try:
            state, actions, reward = simulator.read()
            # except Exception as e:
            #     print(e)

            print(state)
            print('actions: ', actions)
            print(reward)
            print('-----------------------------')

            if not actions:
                break

            action = random.randint(0, len(actions) - 1)
            simulator.write(action)

            # time.sleep(0.1)

        simulator.restart()

    simulator.driver.close()
