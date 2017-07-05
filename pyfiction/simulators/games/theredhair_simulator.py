import random

from pyfiction.games.TheRedHair.the_red_hair import TheRedHair
from pyfiction.simulators.html_simulator import HTMLSimulator


class TheRedHairSimulator(HTMLSimulator):
    def __init__(self, shuffle=True):
        super(TheRedHairSimulator, self).__init__(TheRedHair, shuffle=shuffle)

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
        action.click()

    def read(self):

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
            elif text.lower().startswith('you stay in the bedroom and eventually the parents come back and thank you'):
                reward = 20
            else:
                raise Exception('Game ended and no actions left but an unknown ending reached, cannot assign reward: ',
                                ending)

        elif self.shuffle:
            random.shuffle(self.actions)

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
