import random

from pyfiction.games.HowlingDogs.howling_dogs import HowlingDogs
from pyfiction.simulators.html_simulator import HTMLSimulator


class HowlingDogsSimulator(HTMLSimulator):
    # the maximum number of steps the agent should take before we interrupt him to break infinite cycles
    max_steps = 1000

    # the recommended number of random game walkthroughs for vocabulary initialization
    # should ideally cover all possible states and used words
    initialization_iterations = 1024

    # if the game rewards are in e.g. [-30, 30], set the reward scale to 30 so that the result is in [-1, 1]
    reward_scale = 10

    def __init__(self, shuffle_actions=True):
        super().__init__(HowlingDogs, shuffle_actions=shuffle_actions)
        self.restart()

    def restart(self):
        self.driver.execute_script('state.restart()')
        # self.driver.get('file:///' + self.game.path)
        self.startup_actions()

    def write(self, action_index):
        action = self.actions[action_index]
        self.driver.find_element_by_link_text(action).click()

    def read(self):

        # text is always here:
        text = self.driver.find_element_by_css_selector("div[id='passages']").text

        # actions are always of one of the two class sets below:
        self.actions = [action.text for action in self.driver.find_elements_by_class_name("internalLink")]

        try:
            back = self.driver.find_element_by_class_name("back")
        except:
            back = None
        if back:
            self.actions.append(back.text)

        reward = -0.1

        if not self.actions:
            ending = text.lower()
            if ending.startswith('for everyone who feels that way'):
                reward = 10
            elif ending.startswith('howling dogs by porpentine'):
                reward = -10
            else:
                print('Game ended and no actions left but an unknown ending reached, cannot assign reward: ',
                      ending)

        elif self.shuffle_actions:
            random.shuffle(self.actions)

        return text, self.actions, reward

    def startup_actions(self):
        super(HowlingDogsSimulator, self).startup_actions()


if __name__ == '__main__':
    simulator = HowlingDogsSimulator()

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
            print(actions[action])
            print('-------------------')

        print('****************************************************************')

        simulator.restart()

    simulator.close()
