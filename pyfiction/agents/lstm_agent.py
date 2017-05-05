import logging
import os
import random

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


def preprocess(text, chars='”“'):
    """
    function that removes whitespaces, converts to lowercase, etc.
    :param text: text input 
    :param chars: chars to be removed
    :return: cleaned up text
    """

    # replace newlines with spaces, remove slashes, quotes and double hyphens
    text = text.replace('\\n', ' ').replace('/', '').replace('--', '').replace('‘', '\'').replace('’', '\'').replace(
        '“', '').replace('”', '')
    # remove multiple whitespaces
    text = ' '.join(text.split())

    # optionally remove all given characters
    for c in chars:
        if c in text:
            text = text.replace(c, '')

    # convert to lowercase
    text = text.lower()

    # expand 'm, 's, 're etc.
    text = text. \
        replace('i\'m ', 'i am '). \
        replace('you\'re ', 'you are '). \
        replace('he\'s ', 'he is '). \
        replace('it\'s ', 'it is '). \
        replace('that\'s ', 'that is '). \
        replace('let\'s ', 'let us '). \
        replace('who\'s ', 'who is '). \
        replace('what\'s ', 'what is '). \
        replace('there\'s ', 'there is '). \
        replace('we\'re ', 'we are '). \
        replace('they\'re ', 'they are '). \
        replace('can\'t', 'cannot'). \
        replace('won\'t', 'will not'). \
        replace('n\'t', ' not'). \
        replace('\'d ', ' would '). \
        replace('\'ll ', ' will '). \
        replace('\'ve ', ' have ')

    return text


class LSTMAgent(agent.Agent):
    """
    Basic LSTM agent
    Uses precomputed word embeddings (GloVe)
    """

    def __init__(self, simulator):
        self.experience_sequences_prioritised = []
        random.seed(0)
        np.random.seed(0)

        self.simulator = simulator()
        self.experience = []
        self.experience_sequences = []
        self.embeddings_index = None
        self.word_index = None
        self.model = None
        self.tokenizer = Tokenizer(num_words=10000)

        # parameters
        self.step_cost = -0.01
        self.max_steps = 100  # maximum number of actions in one episode before it is terminated
        self.embeddings_dimensions = 50
        self.embeddings_path = 'glove.6B.' + str(self.embeddings_dimensions) + 'd.txt'
        self.max_words = 8192  # maximum number of unique words to use
        self.state_length = 32  # length of state description in tokens
        self.action_length = 16  # length of action description in tokens

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
        embeddings_index = {}
        f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.embeddings_path))
        for line in f:
            values = line.split()
            word = values[0]
            coefficients = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefficients
        f.close()

        logger.info('GloVe embeddings are using %s word vectors.' % len(embeddings_index))
        self.embeddings_index = embeddings_index

    def create_model(self):
        """
        creates the neural network model using precomputed embeddings applied to the training data
        :return: 
        """
        logger.info('Creating word embeddings.')
        self.create_embeddings()
        logger.info('Transforming words from experience into sequences of embedding indices.')
        self.create_sequences()

        num_words = min(len(self.word_index), self.max_words)
        logger.info('Creating a model based on %s unique tokens in the word index: %s', num_words, self.word_index)

        # indices in word_index start with a 1, 0 is reserved for masking padded value
        embedding_matrix = np.zeros((num_words + 1, self.embeddings_dimensions))

        for word, i in self.word_index.items():
            if i >= self.max_words:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                logger.warning('Word not found in embeddings: %s', word)

        state = Sequential()
        state.add(Embedding(num_words + 1,
                            self.embeddings_dimensions,
                            weights=[embedding_matrix],
                            input_length=self.state_length,
                            mask_zero=True,
                            trainable=False))
        state.add(LSTM(32))
        state.add(Dense(8, activation='tanh'))

        action = Sequential()
        action.add(Embedding(num_words + 1,
                             self.embeddings_dimensions,
                             weights=[embedding_matrix],
                             input_length=self.action_length,
                             mask_zero=True,
                             trainable=False))
        action.add(LSTM(16))
        action.add(Dense(8, activation='tanh'))

        model = Sequential()

        # use Dot instead of Merge (Merge is obsolete soon, but Dot doesn't seem to work)
        model.add(Merge([state, action], mode='dot'))
        # model.add(Dot([state, action], axes=[1, 1]))
        # model.add(Dot(input_shape=[state, action], axes=1))

        model.compile(optimizer=RMSprop(), loss='mse')

        self.model = model

        logger.info('State model: %s', state.summary())
        logger.info('Action model: %s', action.summary())
        logger.info('Complete model: %s', model.summary())

    def create_sequences(self):
        """
        creates sequences of integers (word indices) from lists of strings
        creates experience_sequences (experience with sequences instead of texts)
        """

        state_texts = [x[0] for x in self.experience]
        action_texts = [x[1] for x in self.experience]
        state_next_texts = [x[3] for x in self.experience]
        action_next_texts = [x[5] for x in self.experience]

        logger.info('Fitting tokenizer on action texts and state texts.')

        # vectorize the text samples into a 2D integer tensor
        self.tokenizer.fit_on_texts(action_texts)
        logger.info('Tokenizer after going through actions: %s words - %s', len(self.tokenizer.word_index.items()),
                    self.tokenizer.word_index)
        self.tokenizer.fit_on_texts(state_texts)
        logger.info('Tokenizer after adding states on top of actions: %s words - %s',
                    len(self.tokenizer.word_index.items()),
                    self.tokenizer.word_index)

        self.word_index = self.tokenizer.word_index

        state_sequences = self.tokenizer.texts_to_sequences(state_texts)
        state_next_sequences = self.tokenizer.texts_to_sequences(state_next_texts)
        action_sequences = self.tokenizer.texts_to_sequences(action_texts)
        action_next_sequences = [self.tokenizer.texts_to_sequences(x) for x in action_next_texts]

        states = pad_sequences(state_sequences, maxlen=self.state_length)
        actions = pad_sequences(action_sequences, maxlen=self.action_length)
        states_next = pad_sequences(state_next_sequences, maxlen=self.state_length)
        actions_next = [pad_sequences(x, maxlen=self.action_length) for x in action_next_sequences]

        for i in range(len(self.experience)):

            exp = (states[i], actions[i], self.experience[i][2], states_next[i], self.experience[i][4], actions_next[i])

            self.experience_sequences.append(exp)

            # save a set of unique final states separately for prioritised sampling
            if self.experience[i][4]:
                exp_list = (exp[0].tolist(), exp[1].tolist(), exp[2], exp[3].tolist(), exp[4], exp[5].tolist())
                if exp_list not in self.experience_sequences_prioritised:
                    self.experience_sequences_prioritised.append(exp_list)

        # logger.debug('Experience: %s', self.experience)
        # logger.debug('Experience sequences: %s', self.experience_sequences)
        logger.debug('Unique final experiences - total %s, %s', len(self.experience_sequences_prioritised),
                     self.experience_sequences_prioritised)

        logger.info('All experiences transformed to padded sequences of token indices.')

        logger.info('Shape of the state tensor: %s', states.shape)
        logger.info('Shape of the action tensor: %s', actions.shape)

    def reset(self):
        self.simulator.restart()

    def sample(self, episodes=65536):
        """
        generates training data by repeatedly playing the game using a random policy
        :param episodes: number of games to be played
        :return: doesn't return anything; the data is stored in self.experience
        """
        logger.info('Sampling %s game episodes using random actions.', episodes)
        for i in range(episodes):
            self.play_game(store_experience=True, epsilon=1)
        logger.info('Successfully sampled %s game episodes, got %s experiences.', episodes, len(self.experience))

    def play_game(self, store_experience=True, epsilon=1):

        steps = 0
        total_reward = 0
        # logger.debug('playing new game')
        (state, actions, reward) = self.simulator.read()
        while len(actions) > 0 and steps < self.max_steps:
            # logger.debug('s %s', text)
            action = self.act(state, actions, epsilon)
            # logger.debug('a %s', action)
            self.simulator.write(action)

            last_state = state
            last_action = actions[action]

            (state, actions, reward) = self.simulator.read()

            # override reward from the environment in a non-terminal state (set a small negative value instead of zero)
            if reward == 0 and len(actions) > 0:
                reward = self.step_cost
            if store_experience:
                self.experience.append((preprocess(last_state), preprocess(last_action), reward, preprocess(state),
                                        len(actions) < 1, [preprocess(a) for a in actions]))
                # logger.debug("%s --- %s --- %s", replace(text), actions, reward)

            total_reward += reward
            steps += 1

        self.reset()
        return total_reward

    def train(self, batch_size=64, gamma=0.99, prioritised=False):
        """
        Picks random experiences and trains the model on them
        :param batch_size: number of experiences to be used for training (each is used once)
        :param gamma: discount factor (higher gamma ~ taking future into account more)
        :param prioritised: only sample prioritised experience (final states with higher reward values)
        :return: 
        """

        source = self.experience_sequences

        if prioritised:
            source = self.experience_sequences_prioritised
            logger.debug('sampling prioritised only, %s from %s', batch_size, len(source))

        batches = np.random.choice(len(source), batch_size)

        # logger.debug('Batches: %s', batches)
        # logger.debug('First item: %s', self.experience_sequences[batches[0]])

        states = np.zeros((batch_size, 32))
        actions = np.zeros((batch_size, 16))
        targets = np.zeros((batch_size, 1))

        for i in range(batch_size):
            state, action, reward, state_next, done, actions_next = source[batches[i]]
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

            # logger.debug('trained states: %s', states)
            # logger.debug('trained actions: %s', actions)
            # logger.debug('trained targets: %s', targets)

    def test(self, iterations=16):
        """
        Uses the model to play the game, always picking the action with the highest Q-value.
        :param iterations: Number of games to be played.
        :return: The average score across all iterations.
        """
        score = 0
        for i in range(iterations):
            score += self.play_game(store_experience=True, epsilon=0)

        return score / iterations


def main():
    agent = LSTMAgent(SavingJohnSimulator)
    agent.sample(episodes=8192)
    agent.create_model()

    for i in range(256):
        logger.info('Epoch %s', i)
        # logger.info('Training started')
        agent.train(batch_size=32, prioritised=i < 32)
        # logger.info('Training ended')
        # logger.info('Testing started')
        reward = agent.test(iterations=1)
        logger.info('Average reward: %s', reward)

        # if reward > 19.92:
        #     agent.model.save('lstm-trained' + str(i) + '.hd5')
        #
        # if reward >= 19.94:
        #     r = agent.test(iterations=1)
        #     logger.info('best path found, reward: %s', r)
        #     time.sleep(1)
        #
        # if i % 10 == 0:
        #     agent.model.save('lstm' + str(i) + '.hd5')


if __name__ == "__main__":
    main()
