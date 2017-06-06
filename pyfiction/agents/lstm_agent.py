import logging
import os
import random

import numpy as np
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.layers import LSTM, Dense, Embedding, Dot

from pyfiction.agents import agent

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def preprocess(text, chars='', expand=False):
    """
    function that removes whitespaces, converts to lowercase, etc.
    :param expand: expand 'll, 'm and similar expressions to reduce the number of different tokens
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
    if expand:
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

    def __init__(self, simulator, max_words=8192):

        random.seed(0)
        np.random.seed(0)

        self.simulator = simulator()
        self.experience = []
        self.experience_prioritized = []
        self.unique_endings = []
        self.unique_prioritized_states = []

        self.model = None
        self.tokenizer = Tokenizer(num_words=max_words)  # maximum number of unique words to use

        # visualization
        self.tensorboard = TensorBoard(log_dir='./logs', write_graph=False, write_images=True,
                                       embeddings_freq=1, embeddings_metadata='logs/embeddings.tsv')

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
        state = self.vectorize([text])[0]
        actions = self.vectorize(actions)

        # return an action with maximum Q value
        q_max = -np.math.inf
        best_action = 0
        # logger.debug('q for state %s', state)
        for i in range(len(actions)):
            q = self.q(state, actions[i])
            # logger.debug('q for action "%s" is %s', actions[i], q)
            if q > q_max:
                q_max = q
                best_action = i

        return best_action, q_max

    def create_model(self, embedding_dimensions, lstm_dimensions, dense_dimensions, optimizer, embeddings=None,
                     embeddings_trainable=True):
        """
        creates the neural network model using precomputed embeddings applied to the training data
        :return: 
        """

        num_words = len(self.tokenizer.word_index)
        logger.info('Creating a model based on %s unique tokens in the word index: %s', num_words,
                    self.tokenizer.word_index)

        # create the shared embedding layer (with or without pre-trained weights)
        embedding_shared = None

        if embeddings is None:
            embedding_shared = Embedding(num_words + 1, embedding_dimensions, input_length=None, mask_zero=True,
                                         trainable=embeddings_trainable)
        else:
            logger.info('Importing pre-trained word embeddings.')
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

                    embedding_shared = Embedding(num_words + 1, embedding_dimensions, input_length=None, mask_zero=True,
                                                 trainable=embeddings_trainable, weights=[embedding_matrix])

        input_state = Input(batch_shape=(None, None))
        input_action = Input(batch_shape=(None, None))

        embedding_state = embedding_shared(input_state)
        embedding_action = embedding_shared(input_action)

        lstm_shared = LSTM(lstm_dimensions)
        lstm_state = lstm_shared(embedding_state)
        lstm_action = lstm_shared(embedding_action)

        dense_state = Dense(dense_dimensions, activation='tanh')(lstm_state)
        dense_action = Dense(dense_dimensions, activation='tanh')(lstm_action)

        dot_state_action = Dot(axes=-1, normalize=True)([dense_state, dense_action])

        model = Model(inputs=[input_state, input_action], outputs=dot_state_action)
        model.compile(optimizer=optimizer, loss='mse')

        self.model = model

        print('---------------')
        print('Complete model:')
        model.summary()
        print('---------------')

    def initialize_tokens(self, iterations, max_steps):

        # Temporarily store all experience and use it to get the tokens
        self.play_game(episodes=iterations, max_steps=max_steps, store_experience=True, store_text=True,
                       epsilon=1, verbose=False)

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

        # Store the token information in a text file for embedding visualization
        with open('logs/embeddings.tsv', 'w') as file:
            # write an empty token for the first null embedding
            file.write('EMPTY_EMBEDDING_TOKEN\n')
            for token in list(self.tokenizer.word_index.keys()):
                file.write(token + '\n')

        # Clean text experience data
        self.experience = []

    def vectorize(self, texts, max_len=None):
        """
        converts elements of lists of texts from texts to vectors
        :param texts: list of texts to vectorize
        :param max_len: max vector length, if None, use the longest present sequence
        :return: 
        """
        if not texts:
            return []

        sequences = pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=max_len)

        # tokenizer can return empty sequences for actions such as '...', fix these:
        if not sequences.any():
            return np.asarray([[0]])
        for i in range(len(sequences)):
            if len(sequences[i]) < 1:
                sequences[i] = [0]

        return sequences

    def reset(self):
        self.simulator.restart()

    def play_game(self, max_steps, episodes=1, step_cost=-0.01, store_experience=True, store_text=False, epsilon=1,
                  reward_scale=1, verbose=False):
        """
        Uses the model to play the game.
        :param reward_scale:
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
            experiences = []

            (state, actions, reward) = self.simulator.read()

            while len(actions) > 0 and steps <= max_steps:

                action, q_value = self.act(state, actions, epsilon)

                if verbose:
                    logger.info('State: %s', state)
                    logger.info('Best action: %s, q=%s', actions[action], q_value)

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
                # only if the environment didn't provide a step cost (MoD does, SJ doesn't)
                if reward == 0 and not finished:
                    reward = step_cost

                reward /= reward_scale

                if store_experience:
                    experiences.append((last_state, last_action, reward, state, actions, finished))

                total_reward += reward
                steps += 1

            # TODO - store only finished?
            if store_experience and steps < max_steps:
                for last_state, last_action, reward, state, actions, finished in experiences:
                    self.store_experience(last_state, last_action, reward, state, actions, finished, store_text)

            self.reset()

        if store_text:
            logger.info('Successfully played %s game episodes, got %s new experiences.', episodes, len(self.experience))

        return total_reward * reward_scale / episodes

    def experience_to_sequences(self, state, action, reward, state_next, actions_next, finished):
        """
        Converts a textual form of experience to its equivalent in sequences
        :param state:
        :param action:
        :param reward:
        :param state_next:
        :param actions_next:
        :param finished:
        :return:
        """

        # vectorize the text samples into 2D tensors of word indices
        state_sequence = self.vectorize([state])[0]
        action_sequence = self.vectorize([action])[0]
        state_next_sequence = self.vectorize([state_next])[0]
        actions_next_sequences = self.vectorize(actions_next)

        return state_sequence, action_sequence, reward, state_next_sequence, actions_next_sequences, finished

    def store_experience(self, state, action, reward, state_next, actions_next, finished, store_text_only=False):

        state_text = preprocess(state)
        action_text = preprocess(action)
        state_next_text = preprocess(state_next)
        actions_next_texts = [preprocess(a) for a in actions_next]

        # storing the text version of experience is only necessary prior to fitting the tokenizer
        if store_text_only:
            self.experience.append((state_text, action_text, reward, state_next_text, actions_next_texts, finished))
            return

        experience = self.experience_to_sequences(state_text, action_text, reward, state_next_text, actions_next_texts,
                                                  finished)

        self.experience.append(experience)

        # now return if this experience is not final or doesn't contain a positive reward
        if not finished and reward < 0:
            return

        # store unique final states and/or unique states with a positive reward
        exp_list = (experience[0].tolist(), experience[1].tolist(), experience[2], experience[3].tolist())

        if finished and exp_list not in self.unique_endings:
            self.unique_endings.append(exp_list)
            logger.info('New unique final experience - total %s.\nReward: %s, text: %s',
                        len(self.unique_endings),
                        reward,
                        state_next_text)

        # TODO - prioritize unique positive AND unique cycled/not finished/extremely negative?
        if reward >= 0 and exp_list not in self.unique_prioritized_states:
            self.unique_prioritized_states.append(exp_list)
            logger.info('New unique positive experience - total %s.\nReward: %s, text: %s',
                        len(self.unique_prioritized_states),
                        reward,
                        state_next_text)
            self.experience_prioritized.append(experience)

    def train_offline(self, episodes=1, batch_size=32, gamma=0.99, prioritized=False):
        """
        Picks random experiences and trains the model on them
        :param episodes: number of episodes, in each episode we train batch_size examples
        :param batch_size: number of experiences to be used for training (each is used once in an episode)
        :param gamma: discount factor (higher gamma ~ taking future into account more)
        :param prioritized: only sample prioritized experiences (final states with usually higher reward values)
        :return: 
        """

        source = self.experience

        if prioritized:
            source = self.experience_prioritized
            logger.debug('Sampling prioritized only, %s from %s', batch_size, len(source))

        if len(source) < 1:
            logger.warning('No samples for training available.')
            return

        for x in range(episodes):

            batches = np.random.choice(len(source), batch_size)

            states = [None] * batch_size
            actions = [None] * batch_size
            targets = np.zeros((batch_size, 1))

            for i in range(batch_size):
                state, action, reward, state_next, actions_next, finished = source[batches[i]]
                target = reward

                if not finished:
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

            states = pad_sequences(states)
            actions = pad_sequences(actions)

            self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=1)

    def train_online(self, max_steps, episodes=1024, batch_size=64, gamma=0.99, epsilon=1, epsilon_decay=0.995,
                     prioritized_fraction=0, reward_scale=1, step_cost=-0.01, test_steps=1):
        """
        Trains the model while playing at the same time
        :param reward_scale:
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
                                    epsilon=epsilon, reward_scale=reward_scale)

            logger.info('Episode %s, epsilon = %s, average reward: %s', i, epsilon, reward)

            # Test the agent after each N batches of weight updates
            if ((i + 1) % test_steps) == 0:
                reward = self.play_game(max_steps=max_steps, episodes=1, step_cost=step_cost, store_experience=False,
                                        epsilon=0, reward_scale=reward_scale)
                rewards.append(reward)
                logger.info('Test reward: %s', reward)

            if len(self.experience) < 1:
                return

            batches = np.random.choice(len(self.experience), batch)

            if len(self.experience_prioritized) > 0:
                batches_prioritized = np.random.choice(len(self.experience_prioritized), batch_prioritized)
            else:
                batches_prioritized = np.random.choice(len(self.experience), batch_prioritized)

            states = [None] * batch_size
            actions = [None] * batch_size
            targets = np.zeros((batch_size, 1))

            for b in range(batch_size):

                if b < batch:
                    state, action, reward, state_next, actions_next, finished = self.experience[batches[b]]
                elif len(self.experience_prioritized) > 0:
                    state, action, reward, state_next, actions_next, finished = self.experience_prioritized[
                        batches_prioritized[b - batch]]
                else:
                    state, action, reward, state_next, actions_next, finished = self.experience[
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

            # pad the states and actions so that each sample in this batch has the same size
            states = pad_sequences(states)
            actions = pad_sequences(actions)

            # logger.debug('states %s', states)
            # logger.debug('actions %s', actions)
            # logger.debug('targets %s', targets)

            callbacks = []

            # add a tensorboard callback on the last episode
            if i + 1 == episodes:
                callbacks = [self.tensorboard]

            self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=1,
                           callbacks=callbacks)

            epsilon *= epsilon_decay

        return rewards

    def train_traces(self, max_steps, episodes=256, batch_size=64, gamma=0.99, epsilon=1,
                     epsilon_decay=0.995, reward_scale=1, step_cost=-0.01, test_steps=1):
        """
        Trains the model while playing at the same time
        :param reward_scale:
        :param test_steps: test the agent after each N steps (batches)
        :param epsilon_decay:
        :param epsilon:
        :param step_cost:
        :param episodes:
        :param max_steps:
        :param batch_size: number of experiences to be used for training (each is used once)
        :param gamma:
        :return: rewards
        """

        rewards = []

        for i in range(episodes):

            # clear the experience buffer
            self.experience = []

            # save one trace into the buffer
            reward = self.play_game(max_steps=max_steps, episodes=1, step_cost=step_cost, store_experience=True,
                                    epsilon=epsilon, reward_scale=reward_scale)

            logger.info('Episode %s, epsilon = %s, Train reward: %s', i, epsilon, reward)

            # Test the agent after each N batches of weight updates
            if i == 0 or ((i + 1) % test_steps) == 0:
                reward = self.play_game(max_steps=max_steps, episodes=1, step_cost=step_cost, store_experience=False,
                                        epsilon=0, reward_scale=reward_scale)
                rewards.append(reward)
                logger.info('Test reward: %s', reward)

            last_trace = self.experience

            trace_length = len(last_trace)
            if trace_length < 1:
                logger.warning('No trace available or empty trace sampled, skipping episode %s', i)
                continue

            states = [None] * trace_length
            actions = [None] * trace_length
            targets = np.zeros((trace_length, 1))

            final_reward = None

            j = 0
            reward_decay = 1
            for state, action, reward, _, _, _ in reversed(last_trace):
                if not final_reward:
                    final_reward = reward
                    target = reward
                else:
                    # TODO - adaptive scaling?
                    # TODO target = reward + final_reward * reward_decay (* epsilon)? or combine with a q-step
                    target = reward + final_reward * reward_decay

                # if not finished:
                # get an action with maximum Q value
                # q_max = -np.math.inf
                # for action_next in actions_next:
                #     q = self.q(state_next, action_next)
                #     if q > q_max:
                #         q_max = q
                # target += gamma * q_max

                states[j] = state
                actions[j] = action
                targets[j] = target

                reward_decay *= gamma
                j += 1

            # pad the states and actions so that each sample in this batch has the same size
            states = pad_sequences(states)
            actions = pad_sequences(actions)

            # logger.debug('states %s', states)
            # logger.debug('actions %s', actions)
            # logger.debug('targets %s', targets)

            self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=1,
                           callbacks=[self.tensorboard])

            epsilon *= epsilon_decay

        return rewards

    def q(self, state, action):
        """
        returns the Q-value of a single (state, action) pair
        :param state:
        :param action:
        :return: Q-value estimated by the NN model
        """

        return self.model.predict([state.reshape((1, len(state))), action.reshape((1, len(action)))])[[0]]
