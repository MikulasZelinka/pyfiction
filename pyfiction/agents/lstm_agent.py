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

    def __init__(self, simulator, optimizer=RMSprop(), state_length=64, action_length=16, max_words=8192,
                 max_steps=100):
        self.experience_sequences_prioritised = []
        random.seed(0)
        np.random.seed(0)

        self.simulator = simulator()
        self.experience = []
        self.experience_sequences = []
        self.embeddings_index = None
        self.word_index = None
        self.model = None
        self.tokenizer = Tokenizer(num_words=max_words)  # maximum number of unique words to use
        self.optimizer = optimizer

        # parameters
        self.step_cost = -0.01
        self.max_steps = max_steps  # maximum number of actions in one episode before it is terminated
        self.embeddings_dimensions = 50
        self.embeddings_path = 'glove.6B.' + str(self.embeddings_dimensions) + 'd.txt'
        self.state_length = state_length  # length of state description in tokens
        self.action_length = action_length  # length of action description in tokens

    def act(self, text, actions, epsilon=0):
        """
        returns an action index either randomly or using the model to pick an action with highest Q-value
        :param text: state text
        :param actions: actions to be considered
        :param epsilon: probability of choosing a random action
        :return: index of the picked action and a Q-value
        """
        if (epsilon > 0 and 1 > epsilon > random.random()) or epsilon == 1:
            return random.randint(0, len(actions) - 1), None

        # create sequences from text data
        # text_sequences = pad_sequences(self.tokenizer.texts_to_sequences([text]), maxlen=self.state_length)
        # actions_sequences = pad_sequences(self.tokenizer.texts_to_sequences(actions), maxlen=self.action_length)
        text_sequences = self.vectorize([text], self.state_length)
        actions_sequences = self.vectorize(actions, self.action_length)

        # return an action with maximum Q value
        q_max = -np.math.inf
        best_action = 0
        # logger.debug('q for state %s', text_sequences[0])
        for i in range(len(actions)):
            q = self.q(text_sequences[0], actions_sequences[i])
            # logger.debug('q for action "%s" is %s', actions_sequences[i], q)
            if q > q_max:
                q_max = q
                best_action = i
        # logger.debug('best action is %s', best_action)
        return best_action, q_max

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

        num_words = len(self.word_index)
        logger.info('Creating a model based on %s unique tokens in the word index: %s', num_words, self.word_index)

        # indices in word_index start with a 1, 0 is reserved for masking padded value
        embedding_matrix = np.zeros((num_words + 1, self.embeddings_dimensions))

        for word, i in self.word_index.items():
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
        state.add(LSTM(self.state_length))
        state.add(Dense(8, activation='tanh'))

        action = Sequential()
        action.add(Embedding(num_words + 1,
                             self.embeddings_dimensions,
                             weights=[embedding_matrix],
                             input_length=self.action_length,
                             mask_zero=True,
                             trainable=False))
        action.add(LSTM(self.action_length))
        action.add(Dense(8, activation='tanh'))

        model = Sequential()

        # use Dot instead of Merge (Merge is obsolete soon, but Dot doesn't seem to work)
        model.add(Merge([state, action], mode='dot'))
        # model.add(Dot([state, action], axes=[1, 1]))
        # model.add(Dot(input_shape=[state, action], axes=1))

        model.compile(optimizer=self.optimizer, loss='mse')

        self.model = model

        logger.info('State model: %s', state.summary())
        logger.info('Action model: %s', action.summary())
        logger.info('Complete model: %s', model.summary())

    def vectorize(self, text, max_len):
        """
        converts elements of an experience tuple from texts to vectors 
        :param text: text to vectorize
        :param max_len: max vector length
        :return: 
        """
        return pad_sequences(self.tokenizer.texts_to_sequences(text), maxlen=max_len)

    def create_sequences(self):
        """
        creates sequences of integers (word indices) from lists of strings
        creates experience_sequences (experience with sequences instead of texts)
        """

        state_texts = [x[0] for x in self.experience]
        action_texts = [x[1] for x in self.experience]
        state_next_texts = [x[3] for x in self.experience]
        action_next_texts = [x[4] for x in self.experience]

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

        # check if maximum length parameters make sense for the actual values from the game:
        max_state_length = 0
        for seq in state_sequences:
            if len(seq) > max_state_length:
                max_state_length = len(seq)

        max_action_length = 0
        for seq in action_sequences:
            if len(seq) > max_action_length:
                max_action_length = len(seq)

        logger.info('Max state description length: %s, trimming to max %s', max_state_length, self.state_length)
        logger.info('Max action description length: %s, trimming to max %s', max_action_length, self.action_length)

        if max_state_length < self.state_length:
            self.state_length = max_state_length
            logger.warning('Max found state description length was %s, lowering the max to this value.',
                           self.state_length)

        if max_action_length < self.action_length:
            self.action_length = max_action_length
            logger.warning('Max found action description length was %s, lowering the max to this value.',
                           self.action_length)

        states = pad_sequences(state_sequences, maxlen=self.state_length)
        actions = pad_sequences(action_sequences, maxlen=self.action_length)
        states_next = pad_sequences(state_next_sequences, maxlen=self.state_length)
        actions_next = [pad_sequences(x, maxlen=self.action_length) for x in action_next_sequences]

        for i in range(len(self.experience)):

            exp = (states[i], actions[i], self.experience[i][2], states_next[i], actions_next[i], self.experience[i][5])

            self.experience_sequences.append(exp)

            # save a set of unique final states separately for prioritised sampling
            if exp[5]:
                exp_list = (exp[0].tolist(), exp[1].tolist(), exp[2], exp[3].tolist(), exp[4].tolist(), exp[5])
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

    def play_game(self, episodes=1, store_experience=True, epsilon=1, verbose=False):
        """
        Uses the model to play the game.
        :param episodes: Number of games to be played.
        :param store_experience: Whether to store new experiences for training while playing.
        :param epsilon: Probability of choosing a random action.
        :param verbose: Whether to print states and actions.
        :return: The average score across all episodes.
        """
        total_reward = 0

        for i in range(episodes):

            steps = 0

            (state, actions, reward) = self.simulator.read()
            while len(actions) > 0 and steps <= self.max_steps:

                action, q_value = self.act(state, actions, epsilon)

                if verbose:
                    logger.info('State: %s', state)
                    logger.info('Action: q=%s, %s', q_value, actions[action])

                self.simulator.write(action)

                last_state = state
                last_action = actions[action]

                (state, actions, reward) = self.simulator.read()

                if steps >= self.max_steps:
                    logger.info('Maximum number of steps exceeded, penalising, last state: %s', state)
                    reward -= 100

                # override reward from the environment in a non-terminal state (set a small negative value instead of zero)
                if reward == 0 and len(actions) > 0:
                    reward = self.step_cost
                if store_experience:
                    self.experience.append((preprocess(last_state), preprocess(last_action), reward, preprocess(state),
                                            [preprocess(a) for a in actions], len(actions) < 1))
                    # logger.debug("%s --- %s --- %s", replace(text), actions, reward)

                total_reward += reward
                steps += 1

            self.reset()

        if store_experience:
            logger.info('Successfully played %s game episodes, got %s new experiences.', episodes, len(self.experience))

        return total_reward / episodes

    def train_offline(self, episodes=1, batch_size=32, gamma=0.99, prioritised=False):
        """
        Picks random experiences and trains the model on them
        :param episodes: number of episodes, in each episode we train batch_size examples
        :param batch_size: number of experiences to be used for training (each is used once in an episode)
        :param gamma: discount factor (higher gamma ~ taking future into account more)
        :param prioritised: only sample prioritised experiences (final states with usually higher reward values)
        :return: 
        """

        source = self.experience_sequences

        if prioritised:
            source = self.experience_sequences_prioritised
            logger.debug('sampling prioritised only, %s from %s', batch_size, len(source))

        for x in range(episodes):

            batches = np.random.choice(len(source), batch_size)

            # logger.debug('Batches: %s', batches)
            # logger.debug('First item: %s', self.experience_sequences[batches[0]])

            states = np.zeros((batch_size, self.state_length))
            actions = np.zeros((batch_size, self.action_length))
            targets = np.zeros((batch_size, 1))

            for i in range(batch_size):
                state, action, reward, state_next, actions_next, done = source[batches[i]]
                target = reward

                if not done:
                    # get an action with maximum Q value
                    q_max = -np.math.inf
                    for action_next in actions_next:
                        q = self.q(state_next, action_next)
                        if q > q_max:
                            q_max = q
                    target += gamma * q_max

                states[i] = state
                actions[i] = action
                targets[i] = target

            self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=0)

    def train_online(self, episodes=1024, batch_size=64, gamma=0.99, epsilon=1, epsilon_decay=0.99, prioritised=False):
        """
        Trains the model while playing at the same time
        :param batch_size: number of experiences to be used for training (each is used once)
        :param gamma: discount factor (higher gamma ~ taking future into account more)
        :param prioritised: only sample prioritised experience (final states with higher reward values)
        :return: 
        """

        for i in range(episodes):

            self.reset()
            steps = 0

            (state, actions, reward) = self.simulator.read()
            while len(actions) > 0 and steps <= self.max_steps:
                action, q_value = self.act(state, actions, epsilon)

    def q(self, state, action):
        """
        returns the Q-value of a single (state,action) pair
        :param state:
        :param action:
        :return: Q-value estimated by the NN model
        """
        return self.model.predict([state.reshape((1, self.state_length)), action.reshape((1, self.action_length))])[[0]]


def main():
    """
    Example usage of the LSTMAgent
    :return: 
    """
    agent = LSTMAgent(SavingJohnSimulator)
    agent.sample(episodes=8192)
    agent.create_model()

    for i in range(256):
        logger.info('Epoch %s', i)
        agent.train_offline(batch_size=32, prioritised=i < 32)
        reward = agent.test(iterations=1, verbose=True)
        logger.info('Average reward: %s', reward)

        # agent.model.save('lstm-trained' + str(i) + '.hd5')


if __name__ == "__main__":
    # main()
    pass
