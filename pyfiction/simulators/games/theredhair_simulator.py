import random

from pyfiction.games.TheRedHair.the_red_hair import TheRedHair
from pyfiction.simulators.html_simulator import HTMLSimulator


class TheRedHairSimulator(HTMLSimulator):
    def __init__(self):
        super().__init__(TheRedHair)

    def restart(self):
        # must restart in loop in case the button is not available yet
        while True:
            restarted = True
            try:
                self.driver.find_element_by_id("restart").click()
            except:
                restarted = False
            if restarted:
                break

    def write(self, action):
        self.driver.find_element_by_link_text(action).click()

    def read(self):

        # text is always in the last div:
        text = (self.driver.find_elements_by_css_selector("div")[-1]).text

        # actions are always of one of the two class sets below:
        actions = self.driver.find_elements_by_css_selector("a[class='squiffy-link link-section']")
        # don't fail if there are no actions of the second class
        try:
            actions += self.driver.find_elements_by_css_selector("a[class='squiffy-link link-passage']")
        except:
            pass

        reward = -0.1

        if not actions:
            ending = text.lower()
            if ending.startswith('you lose'):
                reward = -5
            elif ending.endswith('all there is left is a red hair') or ending.endswith(
                'it was the clown statue missing'):
                reward = -10
            elif text.lower().startswith('you stay in the bedroom and eventually the parents come back and thank you'):
                reward = 10
            else:
                raise Exception('Game ended and no actions left but an unknown ending reached, cannot assign reward: ',
                                ending)

        return text, actions, reward


if __name__ == '__main__':
    simulator = TheRedHairSimulator()

    for i in range(16):

        while True:
            text, actions, reward = simulator.read()

            print(text)
            for action in actions:
                print(action.text)
            print(reward)

            if not actions:
                break

            action = actions[random.randint(0, len(actions) - 1)].text
            simulator.write(action)

        simulator.restart()

    simulator.driver.close()
