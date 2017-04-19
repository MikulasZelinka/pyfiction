import logging
import random
import os

import numpy as np
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.layers import LSTM, Dense, Embedding, Dot, Merge
from keras.models import Sequential

from pyfiction.agents import agent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def cleanup(text, chars=''):
    """
    function that removes whitespaces, converts to lowercase, etc.
    :param text: text input to be
    :param chars: chars to be removed
    :return: cleaned up text
    """
    # replace newlines with spaces, remove slashes and double hyphens
    text = text.replace('\\n', ' ').replace('/', '').replace('--', '')
    # remove multiple whitespaces
    text = ' '.join(text.split())

    # optionally remove all given characters
    for c in chars:
        if c in text:
            text = text.replace(c, '')
    return text.lower()


class LSTMAgent(agent.Agent):
    """
    Basic LSTM agent
    Uses precomputed word embeddings (GloVe)
    """

    def __init__(self):
        random.seed(0)
        np.random.seed(0)

        self.simulator = SavingJohnSimulator()
        self.experience = []
        self.experience_sequences = []
        self.embeddings_index = None
        self.word_index = None
        self.model = None
        self.tokenizer = Tokenizer(num_words=10000)

    def act(self, text, actions, epsilon=0):
        """
        returns an action index either randomly or using the model to pick an action with highest Q-value
        :param text: state text
        :param actions: actions to be considered
        :param epsilon: probability of choosing a random action
        :return: index of the picked action
        """
        if (epsilon > 0 and 1 > epsilon > random.random()) or epsilon == 1:
            return random.randint(0, len(actions) - 1)

        # create sequences from text data
        text_sequences = pad_sequences(self.tokenizer.texts_to_sequences([text]), maxlen=32)
        actions_sequences = pad_sequences(self.tokenizer.texts_to_sequences(actions), maxlen=16)

        # return an action with maximum Q value
        q_max = -np.math.inf
        best_action = 0
        for i in range(len(actions)):
            q = self.model.predict([text_sequences[0].reshape((1, 32)), actions_sequences[i].reshape((1, 16))])[[0]]
            # logger.debug('q for action %s is %s', i, q)
            if q > q_max:
                q_max = q
                best_action = i
        # logger.debug('best action is %s', best_action)
        return best_action

    def create_embeddings(self):
        GLOVE_DIR = '/home/myke/'
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        logger.info('Found %s word vectors.' % len(embeddings_index))
        self.embeddings_index = embeddings_index

    def create_model(self):
        """
        creates the neural network model using precomputed embeddings applied to the training data
        :return: 
        """
        self.create_embeddings()
        self.create_sequences()

        num_words = min(10000, len(self.word_index))
        logger.info('Using %s words', num_words)

        # indices in word_index start with a 1 so we will have an empty 0 index
        embedding_matrix = np.zeros((num_words + 1, 50))
        logger.debug('word_index %s', self.word_index)
        for word, i in self.word_index.items():
            if i >= 10000:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        state = Sequential()
        state.add(Embedding(num_words + 1,
                            50,
                            weights=[embedding_matrix],
                            input_length=32,
                            trainable=False))
        state.add(LSTM(64, input_shape=(128, 64)))
        state.add(Dense(8, activation='tanh'))

        action = Sequential()
        action.add(Embedding(num_words + 1,
                             50,
                             weights=[embedding_matrix],
                             input_length=16,
                             trainable=False))
        action.add(LSTM(16, input_shape=(32, 64)))
        action.add(Dense(8, activation='tanh'))

        model = Sequential()

        # use Dot instead of Merge (Merge is obsolete soon, but Dot doesn't seem to work)
        model.add(Merge([state, action], mode='dot'))
        # model.add(Dot([state, action], axes=[1, 1]))
        # model.add(Dot(input_shape=[state, action], axes=1))

        model.compile(optimizer=RMSprop(lr=0.0001), loss='mse')

        self.model = model
        logger.info('Model created: %s', model.summary())

    def create_sequences(self):
        """
        creates sequences of integers (word indices) from lists of strings
        creates experience_sequences (experience with sequences instead of texts)
        """

        state_texts = [x[0] for x in self.experience]
        action_texts = [x[1] for x in self.experience]
        state_next_texts = [x[3] for x in self.experience]
        action_next_texts = [x[5] for x in self.experience]

        # vectorize the text samples into a 2D integer tensor
        self.tokenizer.fit_on_texts(action_texts)
        logger.info('Tokenizer word_index after actions: %s words; %s', len(self.tokenizer.word_index.items()),
                    self.tokenizer.word_index)
        self.tokenizer.fit_on_texts(state_texts)
        logger.info('Tokenizer word_index after states: %s words; %s', len(self.tokenizer.word_index.items()),
                    self.tokenizer.word_index)

        state_sequences = self.tokenizer.texts_to_sequences(state_texts)
        state_next_sequences = self.tokenizer.texts_to_sequences(state_next_texts)
        action_sequences = self.tokenizer.texts_to_sequences(action_texts)
        action_next_sequences = [self.tokenizer.texts_to_sequences(x) for x in action_next_texts]

        self.word_index = self.tokenizer.word_index
        logger.info('Found %s unique tokens.', len(self.word_index))

        states = pad_sequences(state_sequences, maxlen=32)
        actions = pad_sequences(action_sequences, maxlen=16)
        states_next = pad_sequences(state_next_sequences, maxlen=32)
        actions_next = [pad_sequences(x, maxlen=16) for x in action_next_sequences]

        for i in range(len(self.experience)):
            self.experience_sequences.append(
                (states[i], actions[i], self.experience[i][2], states_next[i], self.experience[i][4], actions_next[i]))

        # logger.debug('Experience: %s', self.experience)
        # logger.debug('Experience sequences: %s', self.experience_sequences)

        logger.info('Shape of state tensor: %s', states.shape)
        logger.info('Shape of action tensor: %s', actions.shape)

    def reset(self):
        self.simulator.restart()

    def sample(self, episodes=100000):
        """
        generates training data by repeatedly playing the game using a random policy
        :param episodes: number of games to be played
        :return: doesn't return anything; the data is stored in self.experience
        """
        for i in range(episodes):
            self.play_game(store_experience=True, epsilon=1)
        logger.info('Sampled %s game episodes.', episodes)

    def play_game(self, store_experience=True, epsilon=1):

        total_reward = 0

        (text, actions, reward) = self.simulator.read()
        while len(actions) > 0:

            action = self.act(text, actions, epsilon)
            self.simulator.write(action)

            last_state = text
            last_action = actions[action]

            (text, actions, reward) = self.simulator.read()

            # override reward from the environment (set a small negative value instead of zero)
            if reward == 0:
                reward = -0.01
            if store_experience:
                self.experience.append((cleanup(last_state), cleanup(last_action), reward, cleanup(text),
                                        len(actions) < 1, [cleanup(a) for a in actions]))
                # logger.debug("%s --- %s --- %s", replace(text), actions, reward)

            total_reward += reward

        self.reset()
        return total_reward

    def train(self, batch_size=256, gamma=0.95):
        """
        Picks random experiences and trains the model on them
        :param batch_size: number of experiences to be used for training (each is used once)
        :param gamma: discount factor (higher gamma ~ taking future into account more)
        :return: 
        """
        batches = np.random.choice(len(self.experience_sequences), batch_size)

        # logger.debug('Batches: %s', batches)
        # logger.debug('First item: %s', self.experience_sequences[batches[0]])

        states = np.zeros((batch_size, 32))
        actions = np.zeros((batch_size, 16))
        targets = np.zeros((batch_size, 1))

        for i in range(batch_size):
            state, action, reward, state_next, done, actions_next = self.experience_sequences[batches[i]]
            target = reward

            if not done:
                # get an action with maximum Q value
                q_max = -np.math.inf
                for a in actions_next:
                    q = self.model.predict([state_next.reshape((1, 32)), a.reshape((1, 16))])[[0]]
                    if q > q_max:
                        q_max = q
                target += gamma * q_max

            states[i] = state
            actions[i] = action
            targets[i] = target

            self.model.fit([states, actions], targets, epochs=1, verbose=0)

    def test(self, iterations=256):
        """
        Uses the model to play the game, always picking the action with the highest Q-value.
        :param iterations: Number of games to be played.
        :return: The average score across all iterations.
        """
        score = 0
        for i in range(iterations):
            score += self.play_game(store_experience=False, epsilon=0)

        return score / iterations


def main():
    agent = LSTMAgent()
    agent.sample(1000)
    agent.create_model()

    for i in range(100):
        logger.info('Epoch %s', i)
        logger.info('Training started')
        agent.train()
        logger.info('Training ended')
        logger.info('Testing started')
        logger.info('Average reward: %s', agent.test(iterations=16))
        logger.info('Testing ended')


if __name__ == "__main__":
    main()
