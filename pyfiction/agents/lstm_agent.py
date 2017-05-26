import logging
import os
import random

import numpy as np

from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.layers import LSTM, Dense, Embedding, Dot, Merge
from keras.models import Sequential

from pyfiction.agents import agent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocess(text, chars=''):
    """
    function that removes whitespaces, converts to lowercase, etc.
    :param text: text input 
    :param chars: chars to be removed
    :return: cleaned up text
    """

    # fix bad newlines (replace with spaces), unify quotes
    text = text.replace('\\n', ' ').replace('‘', '\'').replace('’', '\'').replace('”', '"').replace('“', '"')

    # optionally remove all given characters
    for c in chars:
        if c in text:
            text = text.replace(c, '')

    # convert to lowercase
    text = text.lower()

    # expand unambiguous 'm, 't, 're, ... expressions
    text = text. \
        replace('\'m ', ' am '). \
        replace('\'re ', ' are '). \
        replace('won\'t', 'will not'). \
        replace('n\'t', ' not'). \
        replace('\'ll ', ' will '). \
        replace('\'ve ', ' have ')

    return text


def load_embeddings(path):
    embeddings_index = {}
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients
    f.close()

    logger.info('GloVe embeddings are using %s word vectors.' % len(embeddings_index))
    return embeddings_index


class LSTMAgent(agent.Agent):
    """
    Basic Q-learning agent that uses LSTMs on top of word embeddings to estimate Q-values of perceived
     state and action pairs to learn to act optimally
    """

    def __init__(self, simulator, state_length=64, action_length=16, max_words=8192):

        random.seed(0)
        np.random.seed(0)

        self.simulator = simulator()
        self.experience = []
        self.experience_sequences = []
        self.experience_sequences_prioritized = []
        self.unique_endings = []
        self.unique_prioritized_states = []

        self.model = None
        self.tokenizer = Tokenizer(num_words=max_words)  # maximum number of unique words to use

        # parameters
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

    def create_model(self, embedding_dimensions, dense_dimensions, optimizer, embeddings=None,
                     embeddings_trainable=False):
        """
        creates the neural network model using precomputed embeddings applied to the training data
        :return: 
        """

        num_words = len(self.tokenizer.word_index)
        logger.info('Creating a model based on %s unique tokens in the word index: %s', num_words,
                    self.tokenizer.word_index)

        if embeddings is None:
            embedding_state = Embedding(num_words + 1,
                                        embedding_dimensions,
                                        input_length=self.state_length,
                                        mask_zero=True,
                                        trainable=True)
            embedding_action = Embedding(num_words + 1,
                                         embedding_dimensions,
                                         input_length=self.action_length,
                                         mask_zero=True,
                                         trainable=True)
        else:
            logger.info('Creating word embeddings.')
            embeddings_index = load_embeddings(embeddings)

            # indices in word_index start with a 1, 0 is reserved for masking padded value
            embedding_matrix = np.zeros((num_words + 1, embedding_dimensions))

            for word, i in self.tokenizer.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                else:
                    logger.warning('Word not found in embeddings: %s', word)

            embedding_state = Embedding(num_words + 1,
                                        embedding_dimensions,
                                        input_length=self.state_length,
                                        weights=[embedding_matrix],
                                        mask_zero=True,
                                        trainable=embeddings_trainable)
            embedding_action = Embedding(num_words + 1,
                                         embedding_dimensions,
                                         input_length=self.action_length,
                                         weights=[embedding_matrix],
                                         mask_zero=True,
                                         trainable=embeddings_trainable)

        state = Sequential()
        state.add(embedding_state)
        state.add(LSTM(self.state_length))
        state.add(Dense(dense_dimensions, activation='tanh'))

        action = Sequential()
        action.add(embedding_action)
        action.add(LSTM(self.action_length))
        action.add(Dense(dense_dimensions, activation='tanh'))

        model = Sequential()

        # use Dot instead of Merge (Merge is obsolete soon, but Dot doesn't seem to work)
        model.add(Merge([state, action], mode='dot'))
        # model.add(Dot([state, action], axes=[1, 1]))
        # model.add(Dot(input_shape=[state, action], axes=1))

        model.compile(optimizer=optimizer, loss='mse')

        self.model = model

        print('---------------')
        print('State model:')
        state.summary()
        print('Action model:')
        action.summary()
        print('Complete model:')
        model.summary()
        print('---------------')

    def initialize_tokens(self, iterations, max_steps):

        # Temporarily store all experience and use it to get the tokens
        self.play_game(episodes=iterations, max_steps=max_steps, store_experience=True, store_text=True,
                       epsilon=1,
                       verbose=False)

        state_texts = [x[0] for x in self.experience]
        action_texts = [x[1] for x in self.experience]

        logger.info('Fitting tokenizer on action texts and state texts.')

        self.tokenizer.fit_on_texts(action_texts)
        logger.info('Tokenizer after going through actions: %s words - %s', len(self.tokenizer.word_index.items()),
                    self.tokenizer.word_index)
        self.tokenizer.fit_on_texts(state_texts)
        logger.info('Tokenizer after adding states on top of actions: %s words - %s',
                    len(self.tokenizer.word_index.items()),
                    self.tokenizer.word_index)

        # Remove text experience data
        self.experience = None

    def vectorize(self, text, max_len):
        """
        converts elements of an experience lists from texts to vectors 
        :param text: text to vectorize
        :param max_len: max vector length
        :return: 
        """
        return pad_sequences(self.tokenizer.texts_to_sequences(text), maxlen=max_len)

    def reset(self):
        self.simulator.restart()

    def play_game(self, max_steps, episodes=1, step_cost=-0.01, store_experience=True, store_text=False, epsilon=1,
                  verbose=False):
        """
        Uses the model to play the game.
        :param episodes: Number of games to be played.
        :param max_steps: Maximum number of steps (to prevent the agent from cycling)
        :param store_text: Only store text descriptions and not sequences, used for initializing tokenizer
        :param step_cost: Penalty for every step without a reward (usually a small negative value (-0.01), default 0
        :param store_experience: Whether to store new experiences for training while playing.
        :param epsilon: Probability of choosing a random action.
        :param verbose: Whether to print states and actions.
        :return: The average score across all episodes.
        """
        total_reward = 0

        for i in range(episodes):

            steps = 0

            (state, actions, reward) = self.simulator.read()
            while len(actions) > 0 and steps <= max_steps:

                action, q_value = self.act(state, actions, epsilon)

                if verbose:
                    logger.info('State: %s', state)
                    logger.info('Action: q=%s, %s', q_value, actions[action])

                self.simulator.write(action)

                last_state = state
                last_action = actions[action]

                (state, actions, reward) = self.simulator.read()
                finished = len(actions) < 1

                if steps >= max_steps:
                    logger.info('Maximum number of steps exceeded, last state: %s', state)
                    # TODO - penalize?
                    # reward -= 100

                # override reward from the environment in a non-terminal state by applying the step cost
                if reward == 0 and not finished:
                    reward = step_cost
                if store_experience:
                    self.store_experience(last_state, last_action, reward, state, actions, finished, store_text)

                # TODO - somehow normalize reward?
                reward /= 20

                total_reward += reward
                steps += 1

            self.reset()

        if store_text:
            logger.info('Successfully played %s game episodes, got %s new experiences.', episodes, len(self.experience))

        return total_reward / episodes

    def store_experience(self, state, action, reward, state_next, actions_next, finished, store_text_only=False):

        state_text = preprocess(state)
        action_text = preprocess(action)
        state_next_text = preprocess(state_next)
        actions_next_texts = [preprocess(a) for a in actions_next]

        # storing the text version of experience is only necessary prior to fitting the tokenizer
        if store_text_only:
            self.experience.append((state_text, action_text, reward, state_next_text, actions_next_texts, finished))
            return

        # are lists necessary?
        # vectorize the text samples into padded 2D tensors of word indices
        state_sequence = self.vectorize([state_text], self.state_length)
        action_sequence = self.vectorize([action_text], self.action_length)
        state_next_sequence = self.vectorize([state_next_text], self.state_length)
        actions_next_sequences = self.vectorize(actions_next_texts, self.action_length)

        # finished = reward > 0
        finished = len(actions_next_sequences) < 1

        experience = (state_sequence, action_sequence, reward, state_next_sequence, actions_next_sequences, finished)
        self.experience_sequences.append(experience)

        if not finished and reward <= 0:
            return

        # store unique final states and/or unique states with a positive reward
        exp_list = (experience[0].tolist(), experience[1].tolist(), experience[2], experience[3].tolist())

        if finished and exp_list not in self.unique_endings:
            self.unique_endings.append(exp_list)
            logger.info('New unique final experience - total %s.\nReward: %s, text: %s',
                        len(self.unique_endings),
                        reward,
                        state_next_text)

        if reward > 0 and exp_list not in self.unique_prioritized_states:
            self.unique_prioritized_states.append(exp_list)
            logger.info('New unique positive experience - total %s.\nReward: %s, text: %s',
                        len(self.unique_prioritized_states),
                        reward,
                        state_next_text)
            self.experience_sequences_prioritized.append(experience)

    def train_offline(self, episodes=1, batch_size=32, gamma=0.99, prioritized=False):
        """
        Picks random experiences and trains the model on them
        :param episodes: number of episodes, in each episode we train batch_size examples
        :param batch_size: number of experiences to be used for training (each is used once in an episode)
        :param gamma: discount factor (higher gamma ~ taking future into account more)
        :param prioritized: only sample prioritized experiences (final states with usually higher reward values)
        :return: 
        """

        source = self.experience_sequences

        if prioritized:
            source = self.experience_sequences_prioritized
            logger.debug('Sampling prioritized only, %s from %s', batch_size, len(source))

        if len(source) < 1:
            logger.warning('No samples for training available.')
            return

        for x in range(episodes):

            batches = np.random.choice(len(source), batch_size)

            # logger.debug('Batches: %s', batches)
            # logger.debug('First item: %s', self.experience_sequences[batches[0]])

            states = np.zeros((batch_size, self.state_length))
            actions = np.zeros((batch_size, self.action_length))
            targets = np.zeros((batch_size, 1))

            for i in range(batch_size):
                state, action, reward, state_next, actions_next, finished = source[batches[i]]
                target = reward

                # logger.debug('%s actions: %s', len(actions_next), actions_next)
                # logger.debug('complete experience data: %s', source[batches[i]])

                if not finished:
                    # get an action with maximum Q value
                    q_max = -np.math.inf
                    for action_next in actions_next:
                        q = self.q(state_next, action_next)
                        # logger.debug('Q-value: %s', q)
                        if q > q_max:
                            q_max = q
                    target += gamma * q_max

                states[i] = state
                actions[i] = action
                targets[i] = target

            # logger.debug('Targets: %s', targets)

            self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=2)

    def train_online(self, max_steps, episodes=1024, batch_size=64, gamma=0.99, epsilon=1, epsilon_decay=0.995,
                     prioritized_fraction=0.25, step_cost=-0.01, test_steps=16):
        """
        Trains the model while playing at the same time
        :param test_steps: test the agent after each N steps (batches)
        :param epsilon_decay: 
        :param epsilon: 
        :param step_cost: 
        :param episodes: 
        :param max_steps: 
        :param batch_size: number of experiences to be used for training (each is used once)
        :param gamma: discount factor (higher gamma ~ taking future into account more)
        :param prioritized_fraction: only sample prioritized experience (final states with higher reward values)
        :return: rewards
        """

        rewards = []

        batch_prioritized = int(batch_size * prioritized_fraction)
        batch = batch_size - batch_prioritized

        for i in range(episodes):
            reward = self.play_game(max_steps=max_steps, episodes=4, step_cost=step_cost, store_experience=True,
                                    epsilon=epsilon)

            logger.info('Episode %s', i)
            logger.info('Epsilon = %s, average reward: %s', epsilon, reward * 20)

            # Test the agent after each N batches of weight updates
            if i % test_steps == 0:
                reward = self.play_game(max_steps=max_steps, episodes=1, step_cost=step_cost, store_experience=False,
                                        epsilon=0)
                rewards.append(reward * 20)
                logger.info('Test reward: %s', reward * 20)

            states = np.zeros((batch_size, self.state_length))
            actions = np.zeros((batch_size, self.action_length))
            targets = np.zeros((batch_size, 1))

            if len(self.experience_sequences) < 1:
                return

            batches = np.random.choice(len(self.experience_sequences), batch)

            if len(self.experience_sequences_prioritized) > 0:
                batches_prioritized = np.random.choice(len(self.experience_sequences_prioritized), batch_prioritized)
            else:
                batches_prioritized = np.random.choice(len(self.experience_sequences), batch_prioritized)

            for b in range(batch_size):

                if b < batch:
                    state, action, reward, state_next, actions_next, finished = self.experience_sequences[batches[b]]
                elif len(self.experience_sequences_prioritized) > 0:
                    state, action, reward, state_next, actions_next, finished = self.experience_sequences_prioritized[
                        batches_prioritized[b - batch]]
                else:
                    state, action, reward, state_next, actions_next, finished = self.experience_sequences[
                        batches_prioritized[b - batch]]

                target = reward

                if not finished:
                    # get an action with maximum Q value
                    q_max = -np.math.inf
                    for action_next in actions_next:
                        q = self.q(state_next, action_next)
                        if q > q_max:
                            q_max = q
                    target += gamma * q_max

                states[b] = state
                actions[b] = action
                targets[b] = target

            # logger.debug('states %s', states)
            # logger.debug('actions %s', actions)
            # logger.debug('targets %s', targets)

            self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=1)

            epsilon *= epsilon_decay

        return rewards

    def q(self, state, action):
        """
        returns the Q-value of a single (state,action) pair
        :param state:
        :param action:
        :return: Q-value estimated by the NN model
        """
        return self.model.predict([state.reshape((1, self.state_length)), action.reshape((1, self.action_length))])[[0]]
