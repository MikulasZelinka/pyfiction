import random

from pyfiction.games.StarCourt.star_court import StarCourt
from pyfiction.simulators.html_simulator import HTMLSimulator


class StarCourtSimulator(HTMLSimulator):
    def __init__(self, shuffle=True):
        super().__init__(StarCourt, shuffle=shuffle)
        self.actions = None

    def restart(self):
        super().restart()

    def write(self, action_index):
        action = self.actions[action_index][1]
        action.click()

    def read(self):

        # text is always in the the tw-story html tag:
        passage = self.driver.find_element_by_class_name("passage")
        text = passage.text

        # self.actions = [(action.text, action) for action in passage.find_elements_by_class_name("internalLink")]
        self.actions = [(action.text, action) for action in passage.find_elements_by_tag_name("a")]

        reward = -0.1

        # detect endings
        if len(self.actions) == 2 and self.actions[0][0] == "Take survey." and self.actions[1][0] == "Start over.":
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
                # raise Exception('Game ended and no actions left but an unknown ending reached, cannot assign reward: ',
                #                 ending)
                pass

        elif self.shuffle:
            random.shuffle(self.actions)

        return text, [action[0] for action in self.actions], reward

    def close(self):
        self.driver.close()


if __name__ == '__main__':
    simulator = StarCourtSimulator()
    starts = 0
    ends = 0

    for i in range(100):

        while True:

            state, actions, reward = simulator.read()

            if state.startswith('‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌‌The laser gavel pounds.'):
                starts += 1


            print(state)
            print('actions: ', actions)
            print(reward)
            print('-----------------------------')

            if not actions:
                ends += 1
                break

            action = random.randint(0, len(actions) - 1)
            simulator.write(action)

        simulator.restart()

    simulator.driver.close()
    print('starts', starts, 'ends', ends)
