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

        # try:
        self.actions += [(action.text, action) for action in last_state.find_elements_by_tag_name("a") if
                         action.text]

        # except Exception as e:
        #     raise(e)

        # filter the already used actions (workaround for the game bug that causes infinite loops)
        self.actions = [action for action in self.actions if action[0] not in self.actions_history]

        reward = -0.1

        # detect good endings (bad endings have no actions)
        if len(self.actions) == 1 and self.actions[0][0] == "spoilers":
            self.actions = []

        if not self.actions:
            ending = text.lower()

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
                # raise Exception('Game ended and no actions left but an unknown ending reached, cannot assign reward: ',
                #                 ending)
                print('Game ended and no actions left but an unknown ending reached, cannot assign reward: ',
                      ending[:100])



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
