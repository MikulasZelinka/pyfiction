import logging
import random

import numpy as np

np.random.seed(0)

from keras.layers import LSTM, Dense, Activation, Dot
from keras.models import Sequential
from keras.optimizers import RMSprop

from pyfiction.agents import agent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_model():
    # model = Sequential()
    # model.add(LSTM(256, input_shape=(128, 64)))
    # model.add(Dense(16))
    # model.add(Activation('softmax'))
    # optimizer = RMSprop(lr=0.01)
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # return model

    state = Sequential()
    state.add(LSTM(64, input_shape=(128, 64)))
    state.add(Dense(8))

    action = Sequential()
    action.add(LSTM(16, input_shape=(32, 64)))
    action.add(Dense(4))

    model = Dot([state, action], mode='dot', dot_axes=(1, 1))


def replace(text, chars=''):
    # replace newlines with spaces, remove slashes and double hyphens
    text = text.replace('\\n', ' ').replace('/', '').replace('--', '')
    # remove multiple whitespaces
    text = ' '.join(text.split())

    # optionally remove all given characters
    for c in chars:
        if c in text:
            text = text.replace(c, '')
    return text


class LSTMAgent(agent.Agent):
    """
    Basic LSTM agent
    """

    def __init__(self):
        random.seed(0)
        self.experience = []
        self.simulator = SavingJohnSimulator()
        self.model = create_model()
        self.gamma = 0.9

    def act(self, text, actions, reward):
        return random.randint(0, len(actions) - 1)

    def reset(self):
        self.simulator.restart()

    def get_experience(self, episodes=100000):
        """
        :param episodes: number of games to be played
        :return: experience replay
        """
        for i in range(episodes):
            self.play_game()
            self.reset()

    def play_game(self):
        (text, actions, reward) = self.simulator.read()
        while len(actions) > 0:

            action = self.act(text, actions, reward)
            self.simulator.write(action)

            last_state = text
            last_action = actions[action]

            (text, actions, reward) = self.simulator.read()

            # override reward from the environment (set small negative instead of zero)
            if reward == 0:
                reward = -0.01

            self.experience.append((replace(last_state), replace(last_action), reward, replace(text), len(actions) < 1))
            # logger.debug("%s --- %s --- %s", replace(text), actions, reward)

    def train(self, batch_size=16):
        batches = np.random.choice(len(self.experience), batch_size)

        for i in batches:
            state, action, reward, next_state, done = self.experience[i]
            target = reward
            if not done:
                target = reward + self.gamma * np.argmax(self.model.predict(next_state)[0])

        pass

    def test(self, iterations=256):

        pass


def main():
    agent = LSTMAgent()
    agent.get_experience(10)

    logger.debug(agent.experience)

    for i in range(1000):
        agent.train()
        agent.test()


if __name__ == "__main__":
    main()
