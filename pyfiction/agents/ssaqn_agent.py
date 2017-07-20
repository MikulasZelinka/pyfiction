import logging
import os
import random

import datetime
from collections import deque

import numpy as np
import re
from keras import Input
from keras.callbacks import TensorBoard
from keras.engine import Model

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.layers import LSTM, Dense, Embedding, Dot
from selenium.common.exceptions import NoSuchElementException

from pyfiction.agents import agent
from pyfiction.simulators.simulator import Simulator, UnknownEndingException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Source: https://stackoverflow.com/questions/34968722/softmax-function-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess(text, chars='', remove_all_special=True, expand=True, split_numbers=True):
    """
    function that removes whitespaces, converts to lowercase, etc.
    :param split_numbers: split 45787 to 4 5 7 8 7
    :param remove_all_special: remove all characters but  alpha-numerical, spaces, hyphens, quotes
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

    # remove all characters except alphanum, spaces and - ' "
    if remove_all_special:
        text = re.sub('[^ \-\sA-Za-z0-9"\']+', ' ', text)

    # split numbers into digits to avoid infinite vocabulary size if random numbers are present:
    if split_numbers:
        text = re.sub('[0-9]', ' \g<0> ', text)

    # expand unambiguous 'm, 't, 're, ... expressions
    if expand:
        text = text. \
            replace('\'m ', ' am '). \
            replace('\'re ', ' are '). \
            replace('won\'t', 'will not'). \
            replace('n\'t', ' not'). \
            replace('\'ll ', ' will '). \
            replace('\'ve ', ' have '). \
            replace('\'s', ' \'s')

    return text


def load_embeddings(path):
    """
    loads embeddings from a file and their their index (a dictionary of words with coefficients)
    tested on GloVe
    :param path: path to the embedding file
    :return:
    """
    embeddings_index = {}
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients
    f.close()

    logger.info('Imported embeddings are using %s word vectors.' % len(embeddings_index))
    return embeddings_index


class SSAQNAgent(agent.Agent):
    """
    Siamese State-Action Q-Network agent
    A type Q-learning agent that uses shared LSTMs on top of shared word embeddings to estimate Q-values of perceived
        state and action pairs to learn to act optimally.

    Architecture of the q(s, a) NN estimator:
        Embedding(state), Embedding(action) - LSTM(s), LSTM(a) - DenseState(s), DenseAction(a) - Dot(s, a)

    Features:
        - Embedding and LSTM layers are shared between states and actions
        - supports loading pre-trained word embeddings (such as GloVe or word2vec)
        - experience-replay with prioritized sampling (experiences with positive rewards are prioritized)
        - uses intrinsic motivation in the form of penalizing q(s, a) if 'a' was already chosen in 's'
          for action selection

    This agent class is universal and it should be possible to apply it to different games in the same way
    """

    def __init__(self, train_simulators, test_simulators=None, log_folder='logs', max_words=8192):
        """

        :param train_simulators:
            A single simulator or a list of game simulators used for training
        :param test_simulators:
            A single simulator or a list of game simulators used for testing; if None, train simulators are used
        :param max_words: maximum vocabulary size
        """

        random.seed(0)
        np.random.seed(0)

        if isinstance(train_simulators, Simulator):
            train_simulators = [train_simulators]

        if isinstance(test_simulators, Simulator):
            test_simulators = [test_simulators]

        self.train_simulators = train_simulators
        self.test_simulators = test_simulators if test_simulators else train_simulators

        # a reference to the currently used simulator, varies in time, used by various agent functions
        self.simulator = None

        self.experience = []

        # for prioritized sampling of positive experiences
        self.prioritized_experiences_queue = deque(maxlen=64)
        # for detection of unique experiences (stores lists instead of data used for learning):
        self.unique_prioritized_experiences_queue = deque(maxlen=64)

        self.state_action_history = None

        self.model = None

        # subsets of the model used for partial forward passes:
        # model = model_dot_state_action(model_state, model_action)
        self.model_state = None
        self.model_action = None
        self.model_dot_state_action = None

        self.tokenizer = Tokenizer(num_words=max_words)  # maximum number of unique words to use

        self.log_folder = log_folder
        os.makedirs(log_folder, exist_ok=True)

        # visualization
        self.tensorboard = TensorBoard(log_dir='./logs', write_graph=False, write_images=True,
                                       embeddings_freq=1, embeddings_metadata=log_folder + '/embeddings.tsv')

    def act(self, state, actions, epsilon=0):
        """
        returns an action index either randomly or using the model to pick an action with highest Q-value
        :param state: state text
        :param actions: actions to be considered
        :param epsilon: probability of choosing a random action
        :return: index of the picked action and a Q-value (None if random)
        """
        if epsilon == 1 or (epsilon > 0 and 1 > epsilon > random.random()):
            return random.randint(0, len(actions) - 1), None

        state = self.vectorize([state])[0]
        actions = self.vectorize(actions)

        # return an action with maximum Q value
        return self.q_precomputed_state(state, actions, softmax_selection=False, penalize_history=True)

    def clear_experience(self):
        """
        Clears all sampling experience of the agent
        :return:
        """
        self.experience = []
        self.prioritized_experiences_queue.clear()
        self.unique_prioritized_experiences_queue.clear()

    def create_model(self, embedding_dimensions, lstm_dimensions, dense_dimensions, optimizer, embeddings=None,
                     embeddings_trainable=True):
        """
        creates the neural network model, optionally using precomputed embeddings applied to the training data
        :return: 
        """

        num_words = len(self.tokenizer.word_index)
        logger.info('Creating a model based on %s unique tokens.', num_words)

        # create the shared embedding layer (with or without pre-trained weights)
        embedding_shared = None

        if embeddings is None:
            embedding_shared = Embedding(num_words + 1, embedding_dimensions, input_length=None, mask_zero=True,
                                         trainable=embeddings_trainable, name="embedding_shared")
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
                                                 trainable=embeddings_trainable, weights=[embedding_matrix],
                                                 name="embedding_shared")

        input_state = Input(batch_shape=(None, None), name="input_state")
        input_action = Input(batch_shape=(None, None), name="input_action")

        embedding_state = embedding_shared(input_state)
        embedding_action = embedding_shared(input_action)

        lstm_shared = LSTM(lstm_dimensions, name="lstm_shared")
        lstm_state = lstm_shared(embedding_state)
        lstm_action = lstm_shared(embedding_action)

        dense_state = Dense(dense_dimensions, activation='tanh', name="dense_state")(lstm_state)
        dense_action = Dense(dense_dimensions, activation='tanh', name="dense_action")(lstm_action)

        model_state = Model(inputs=input_state, outputs=dense_state, name="state")
        model_action = Model(inputs=input_action, outputs=dense_action, name="action")

        self.model_state = model_state
        self.model_action = model_action

        input_dot_state = Input(shape=(dense_dimensions,))
        input_dot_action = Input(shape=(dense_dimensions,))
        dot_state_action = Dot(axes=-1, normalize=True, name="dot_state_action")([input_dot_state, input_dot_action])

        model_dot_state_action = Model(inputs=[input_dot_state, input_dot_action], outputs=dot_state_action,
                                       name="dot_state_action")
        self.model_dot_state_action = model_dot_state_action

        model = Model(inputs=[model_state.input, model_action.input],
                      outputs=model_dot_state_action([model_state.output, model_action.output]),
                      name="model")
        model.compile(optimizer=optimizer, loss='mse')

        self.model = model

        print('---------------')
        print('Complete model:')
        model.summary()
        print('---------------')

    def initialize_tokens(self, vocabulary=None):
        """
        Initialize the agent's vocabulary by either randomly sampling the simulators or by specifying a list of words
        :param vocabulary: path to the file with list of tokens (one per line)
        :return:
        """

        if vocabulary:
            logger.info('Initializing tokens by loading them from %s', vocabulary)

            try:
                with open(vocabulary, "r") as f:
                    words = f.readlines()

                self.tokenizer.fit_on_texts(words)

                logger.info('Tokenizer: %s words - %s', len(self.tokenizer.word_index.items()),
                            self.tokenizer.word_index)
                return
            except IOError as e:
                logger.warning('Could not find the specified vocabulary file %s: %s; Sampling new data instead',
                               vocabulary, e)

        logger.info('Initializing tokens by playing the game randomly with all train and test simulators')

        for simulator in list(set(self.train_simulators + self.test_simulators)):
            logger.info('Playing %s randomly %s times', simulator.game.name, simulator.initialization_iterations)

            # Temporarily store all experience and use it to get the tokens, use each simulator once
            self.play_game(episodes=simulator.initialization_iterations, store_experience=True, initialize_only=True,
                           epsilon=1, simulators=[simulator])

        logger.info('Successfully sampled all games, got total %s experiences', len(self.experience))

        state_texts = [x[0] for x in self.experience]
        action_texts = [x[1] for x in self.experience]

        logger.info('Fitting tokenizer on action texts and state texts.')

        self.tokenizer.fit_on_texts(action_texts)
        logger.info('Tokenizer after fitting on actions: %s words - %s', len(self.tokenizer.word_index.items()),
                    self.tokenizer.word_index)
        self.tokenizer.fit_on_texts(state_texts)
        logger.info('Tokenizer after fitting states on top of actions: %s words - %s',
                    len(self.tokenizer.word_index.items()),
                    self.tokenizer.word_index)

        # Store the sampled vocabulary so that we don't have to always sample the game randomly
        with open('vocabulary.txt', 'w') as f:
            f.writelines([word + '\n' for word in self.tokenizer.word_index.keys()])
        logger.info('Saved the vocabulary to vocabulary.txt')

        # Store the token information in a text file for embedding visualization in tensorboard
        with open(self.log_folder + '/embeddings.tsv', 'wb') as f:
            # write an empty token for the first null embedding
            f.write(b'EMPTY_EMBEDDING_TOKEN\n')
            for token in list(self.tokenizer.word_index.keys()):
                f.write(token.encode('utf-8') + b'\n')
            logger.info('Saved the embedding dictionary to logs/embeddings.tsv')

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

    def play_game(self, simulators, episodes=1, store_experience=True, initialize_only=False, epsilon=1):
        """
        Uses the model to play the game. Each simulator is used to play the game exactly episodes times.
        :param simulators: a list of games (their simulators) to play the game with (more simulators = more games)
        :param episodes: Number of games to be played.
        :param initialize_only: Only store text descriptions and not sequences, used for initializing tokenizer
        :param store_experience: Whether to store new experiences for training while playing.
        :param epsilon: Probability of choosing a random action.
        :return: The list of average rewards for all simulators across all episodes.
        """

        rewards = []

        # used for variable episode count per simulator
        episodes_per_simulator = episodes

        for s in range(len(simulators)):

            simulator = simulators[s]

            # set the reference to the current simulator so that other methods called from here can access it too
            self.simulator = simulator

            simulator_rewards = []

            if isinstance(episodes, list):
                episodes_per_simulator = episodes[s]

            for i in range(episodes_per_simulator):

                steps = 0
                experiences = []

                self.reset_history()

                try:
                    (state, actions, reward) = self.simulator.read()
                except (UnknownEndingException, NoSuchElementException, IndexError) as e:
                    logger.error('LSTM Agent simulator read error: %s', e)
                    logger.warning('Interrupting the current episode, not assigning any reward')
                    self.simulator.restart()
                    continue

                state = preprocess(state)
                actions = [preprocess(a) for a in actions]

                episode_reward = 0

                # indicates whether a simulator read error has occurred
                error = False

                while len(actions) > 0 and steps <= self.simulator.max_steps:

                    action, q_value = self.act(state, actions, epsilon)

                    logger.debug('State: %s', state)
                    logger.debug('Best action: %s, q=%s', actions[action], q_value)

                    # only save to history if tokenizer is already initialized == not storing text data
                    if not initialize_only:
                        self.add_to_history(self.vectorize([state])[0], self.vectorize([actions[action]])[0])

                    self.simulator.write(action)

                    last_state = state
                    last_action = actions[action]

                    error = False

                    try:
                        (state, actions, reward) = self.simulator.read()
                    except (UnknownEndingException, NoSuchElementException, IndexError) as e:
                        logger.error('LSTM Agent simulator read error: %s', e)
                        logger.warning('Interrupting the current episode, not assigning any reward')
                        # simulator_rewards.append(None)
                        error = True
                        break

                    state = preprocess(state)
                    actions = [preprocess(a) for a in actions]

                    finished = len(actions) < 1

                    if steps >= self.simulator.max_steps:
                        logger.info('Maximum number of steps exceeded, last state: %s...', state[:64])
                        # TODO - penalize?
                        # reward -= 100

                    # scale the reward from [-reward_scale, reward_scale] to [-1, 1]
                    reward /= self.simulator.reward_scale

                    if store_experience:
                        experiences.append((last_state, last_action, reward, state, actions, finished))

                    episode_reward += reward
                    steps += 1

                if not error:
                    # store only episodes that did not exceed max steps or if still getting tokens
                    if store_experience and (steps < self.simulator.max_steps or initialize_only):
                        for last_state, last_action, reward, state, actions, finished in experiences:
                            self.store_experience(last_state, last_action, reward,
                                                  state, actions, finished, initialize_only)

                    simulator_rewards.append(episode_reward * self.simulator.reward_scale)

                self.simulator.restart()

            rewards.append(simulator_rewards)

        return rewards

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

    def store_experience(self, state_text, action_text, reward, state_next_text, actions_next_texts, finished,
                         store_text_only=False):
        """
        Stores the supplied tuple into the experience replay memory D.
        :param state_text:
        :param action_text:
        :param reward:
        :param state_next_text:
        :param actions_next_texts:
        :param finished:
        :param store_text_only:
        :return:
        """

        # storing the text version of experience is only necessary prior to fitting the tokenizer
        if store_text_only:
            self.experience.append((state_text, action_text, reward, state_next_text, actions_next_texts, finished))
            return

        experience = self.experience_to_sequences(state_text, action_text, reward, state_next_text, actions_next_texts,
                                                  finished)

        self.experience.append(experience)

        # now return if this experience doesn't contain a positive reward
        if reward <= 0:
            return

        # change to a hashable type for the 'in' operator below:
        exp_list = (experience[0].tolist(), experience[1].tolist(), experience[2], experience[3].tolist())

        # store unique positive experiences
        # TODO - prioritize unique positive AND unique cycled/not finished/extremely negative?
        if reward > 0 and (exp_list not in self.unique_prioritized_experiences_queue):
            self.unique_prioritized_experiences_queue.append(exp_list)
            self.prioritized_experiences_queue.append(experience)
            logger.info('New unique positive experience - total {}.\nReward: {:.1f}, state: {}...'.format(
                len(self.prioritized_experiences_queue),
                reward * self.simulator.reward_scale,
                state_next_text[:64]))

    def train_online(self, episodes=256, batch_size=256, gamma=0.95, epsilon=1, epsilon_decay=0.99,
                     prioritized_fraction=0, test_interval=1, test_steps=1, checkpoint_steps=128, log_prefix=''):
        """
        Trains the model while playing at the same time
        :param log_prefix: file prefix for logs
        :param test_steps: test the model each N steps on the test simulators
        :param checkpoint_steps: save the model each N steps
        :param test_interval: test the agent after each N steps (batches)
        :param epsilon_decay: rate at which epsilon decays (in each episodes, epsilon *= epsilon_decay)
        :param epsilon: probability of choosing a random action
        :param episodes: the maximum number of episodes
        :param batch_size: number of experiences to be used for training (each is used once)
        :param gamma: discount factor (higher gamma ~ taking future into account more)
        :param prioritized_fraction: only sample prioritized experience (final states with higher reward values)
        :return:
        """

        train_rewards_history = []
        test_rewards_history = []

        # batch_prioritized is the number of prioritized samples to get
        batch_prioritized = int(batch_size * prioritized_fraction)
        # batch is the number of any samples to get
        batch = batch_size - batch_prioritized

        for i in range(episodes):

            logger.info('\n------------------------------------------\nEpisode {}, epsilon = {:.4f}'.format(i, epsilon))

            # Epsilon-greedy train sampling and experience saving (epsilon usually decreasing over time):
            train_rewards = self.play_game(episodes=1, store_experience=True, epsilon=epsilon,
                                           simulators=self.train_simulators)
            train_rewards_history.append(train_rewards)
            logger.info(
                "Train rewards: " +
                ", ".join(
                    " ".join(x for x in ['{:.1f}'.format(reward) for reward in simulator_rewards])
                    for simulator_rewards in train_rewards)
            )

            # Test the agent after each test_steps episodes with a zero epsilon
            if ((i + 1) % test_interval) == 0:
                test_rewards = self.play_game(episodes=test_steps, store_experience=False, epsilon=0,
                                              simulators=self.test_simulators)
                test_rewards_history.append(test_rewards)
                logger.info(
                    "Test rewards: " +
                    ", ".join(
                        " ".join(x for x in ['{:.1f}'.format(reward) for reward in simulator_rewards])
                        for simulator_rewards in test_rewards)
                )

            if len(self.experience) < 1:
                return

            batches = np.random.choice(len(self.experience), batch)

            if len(self.prioritized_experiences_queue) > 0:
                batches_prioritized = np.random.choice(len(self.prioritized_experiences_queue), batch_prioritized)
            else:
                batches_prioritized = np.random.choice(len(self.experience), batch_prioritized)

            states = [None] * batch_size
            actions = [None] * batch_size
            targets = np.zeros((batch_size, 1))

            for b in range(batch_size):

                # non-prioritized data:
                if b < batch:
                    state, action, reward, state_next, actions_next, finished = self.experience[batches[b]]
                # prioritized data (if there are any)
                elif len(self.prioritized_experiences_queue) > 0:
                    state, action, reward, state_next, actions_next, finished = self.prioritized_experiences_queue[
                        batches_prioritized[b - batch]]
                # get non-prioritized if there are no prioritized
                else:
                    state, action, reward, state_next, actions_next, finished = self.experience[
                        batches_prioritized[b - batch]]

                _, current_q = self.q_precomputed_state(state, [action], penalize_history=False)
                alpha = 1

                target = current_q + alpha * (reward - current_q)

                if not finished:
                    # get an action with maximum Q value
                    _, q_max = self.q_precomputed_state(state_next, actions_next, penalize_history=False)
                    target += alpha * gamma * q_max

                states[b] = state
                actions[b] = action
                targets[b] = target

            # pad the states and actions so that each sample in this batch has the same size
            states = pad_sequences(states)
            actions = pad_sequences(actions)

            logger.debug('states %s', states)
            logger.debug('actions %s', actions)
            logger.debug('targets %s', targets)

            callbacks = []

            # add a tensorboard callback on the last episode
            if i + 1 == episodes:
                callbacks = [self.tensorboard]

            self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=0,
                           callbacks=callbacks)

            epsilon *= epsilon_decay

            # every checkpoint_steps, write down train and test reward history and the model object
            if ((i + 1) % checkpoint_steps) == 0:

                file_name = 'ep' + str(i) + '_' + datetime.datetime.now().strftime('%m-%d-%H_%M_%S')

                with open(self.log_folder + '/' + log_prefix + '_train_' + file_name + '.txt', 'w') as file:
                    for simulator_rewards in train_rewards_history:
                        for rewards in simulator_rewards:
                            for reward in rewards:
                                file.write('{:.1f}'.format(reward) + ' ')
                            file.write(',')
                        file.write('\n')

                with open(self.log_folder + '/' + log_prefix + '_test_' + file_name + '.txt', 'w') as file:
                    for simulator_rewards in test_rewards_history:
                        for rewards in simulator_rewards:
                            for reward in rewards:
                                file.write('{:.1f}'.format(reward) + ' ')
                            file.write(',')
                        file.write('\n')

                # save the model
                self.model.save(self.log_folder + '/' + log_prefix + file_name + '.h5')

        return

    def q_precomputed_state(self, state, actions, softmax_selection=False, penalize_history=False):
        """
        returns the Q-value of a single (state, action) pair
        :param state: state text data (embedding index)
        :param actions: actions text data (embedding index)
        :param softmax_selection: apply random softmax selection
        :param penalize_history: account for history in this episode - penalize already visited (state, action) tuples
        :return: (best action index, best action Q-value estimated by the NN model)
        """

        state_dense = self.model_state.predict([state.reshape((1, len(state)))])[0]

        q_max = -np.math.inf
        best_action = 0

        q_values = np.zeros(len(actions))

        logger.debug('q for state %s', state)
        for i in range(len(actions)):

            action = actions[i]
            action_dense = self.model_action.predict([action.reshape((1, len(action)))])[0]

            q = self.model_dot_state_action.predict(
                [state_dense.reshape((1, len(state_dense))), action_dense.reshape((1, len(action_dense)))])[0][0]

            if penalize_history:
                # apply intrinsic motivation (penalize already visited (state, action) tuples)
                history = self.get_history(state, action)
                if history:
                    # q is a cosine similarity (dot product of normalized vectors), ergo q is in [-1; 1]
                    # map it to [0; 1]
                    q = (q + 1) / 2

                    q = q ** (history + 1)

                    # map q back to [-1; 1]
                    q = (q * 2) - 1

            logger.debug('q for action %s is %s', action, q)

            q_values[i] = q

            if q > q_max:
                q_max = q
                best_action = i

        if softmax_selection:
            probabilities = softmax(q_values)
            x = random.random()
            for i in range(len(actions)):
                if x <= probabilities[i]:
                    return i, q_values[i]
                x -= probabilities[i]

        return best_action, q_max

    def add_to_history(self, state, action):
        """
        Adds a state, action pair to the history of the current episode, or increases its counter if already present
        :param state:
        :param action:
        :return:
        """

        state = tuple(np.trim_zeros(state, 'f'))
        action = tuple(np.trim_zeros(action, 'f'))

        if (state, action) in self.state_action_history:
            self.state_action_history[(state, action)] += 1
        else:
            self.state_action_history[(state, action)] = 1

    def get_history(self, state, action):
        """
        :param state:
        :param action:
        :return: h(s,a), i.e. the number of times given action was selected in given state in the current episode
        """

        state = tuple(np.trim_zeros(state, 'f'))
        action = tuple(np.trim_zeros(action, 'f'))

        if (state, action) in self.state_action_history:
            return self.state_action_history[(state, action)]
        return 0

    def reset_history(self):
        """
        resets the history; called every time a game episode ends
        :return:
        """
        self.state_action_history = {}

    def q(self, state_text, action_text):
        """
        Computes the q-value of a state text and an action text; useful for testing trained agents
        :param state_text:
        :param action_text:
        :return:
        """
        state = self.vectorize([preprocess(state_text)])[0]
        action = self.vectorize([preprocess(action_text)])[0]

        print('State embedding indices:', state)
        print('State embedding tokens:',
              [list(self.tokenizer.word_index.keys())[list(self.tokenizer.word_index.values()).index(x)]
               if x in self.tokenizer.word_index.values() else ''
               for x in state])

        print('Action embedding indices:', action)
        print('Action embedding tokens:',
              [list(self.tokenizer.word_index.keys())[list(self.tokenizer.word_index.values()).index(x)]
               if x in self.tokenizer.word_index.values() else ''
               for x in action])

        return self.q_precomputed_state(state, [action])[1]


        # offline and per-trace training have not been updated since possibly breaking changes in the agent class
        # TODO test and readd these
        # def train_traces(self, max_steps, episodes=256, batch_size=64, gamma=0.99, epsilon=1,
        #                  epsilon_decay=0.995, reward_scale=1, step_cost=-0.1, test_steps=1):
        #     """
        #     Trains the model while playing at the same time
        #     :param reward_scale:
        #     :param test_steps: test the agent after each N steps (batches)
        #     :param epsilon_decay:
        #     :param epsilon:
        #     :param step_cost:
        #     :param episodes:
        #     :param max_steps:
        #     :param batch_size: number of experiences to be used for training (each is used once)
        #     :param gamma:
        #     :return: rewards
        #     """
        #
        #     rewards = []
        #
        #     for i in range(episodes):
        #
        #         # clear the experience buffer
        #         self.experience = []
        #
        #         # save one trace into the buffer
        #         reward = self.play_game(max_steps=max_steps, episodes=1, step_cost=step_cost, store_experience=True,
        #                                 epsilon=epsilon, reward_scale=reward_scale)
        #
        #         logger.info('Episode %s, epsilon = %s, Train reward: %s', i, epsilon, reward)
        #
        #         # Test the agent after each N batches of weight updates
        #         if i == 0 or ((i + 1) % test_steps) == 0:
        #             reward = self.play_game(max_steps=max_steps, episodes=1, step_cost=step_cost, store_experience=False,
        #                                     epsilon=0, reward_scale=reward_scale)
        #             rewards.append(reward)
        #             logger.info('Test reward: %s', reward)
        #
        #         last_trace = self.experience
        #
        #         trace_length = len(last_trace)
        #         if trace_length < 1:
        #             logger.warning('No trace available or empty trace sampled, skipping episode %s', i)
        #             continue
        #
        #         states = [None] * trace_length
        #         actions = [None] * trace_length
        #         targets = np.zeros((trace_length, 1))
        #
        #         final_reward = None
        #
        #         j = 0
        #         reward_decay = 1
        #         for state, action, reward, _, _, _ in reversed(last_trace):
        #             if not final_reward:
        #                 final_reward = reward
        #                 target = reward
        #             else:
        #                 # TODO - adaptive scaling?
        #                 # TODO target = reward + final_reward * reward_decay (* epsilon)? or combine with a q-step
        #                 target = reward + final_reward * reward_decay
        #
        #             # if not finished:
        #             # get an action with maximum Q value
        #             # q_max = -np.math.inf
        #             # for action_next in actions_next:
        #             #     q = self.q(state_next, action_next)
        #             #     if q > q_max:
        #             #         q_max = q
        #             # target += gamma * q_max
        #
        #             states[j] = state
        #             actions[j] = action
        #             targets[j] = target
        #
        #             reward_decay *= gamma
        #             j += 1
        #
        #         # pad the states and actions so that each sample in this batch has the same size
        #         states = pad_sequences(states)
        #         actions = pad_sequences(actions)
        #
        #         logger.debug('states %s', states)
        #         logger.debug('actions %s', actions)
        #         logger.debug('targets %s', targets)
        #
        #         self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=1,
        #                        callbacks=[self.tensorboard])
        #
        #         epsilon *= epsilon_decay
        #
        #     return rewards

        # def train_offline(self, episodes=1, batch_size=32, gamma=0.99, prioritized=False):
        #     """
        #     Picks random experiences and trains the model on them
        #     :param episodes: number of episodes, in each episode we train batch_size examples
        #     :param batch_size: number of experiences to be used for training (each is used once in an episode)
        #     :param gamma: discount factor (higher gamma ~ taking future into account more)
        #     :param prioritized: only sample prioritized experiences (final states with usually higher reward values)
        #     :return:
        #     """
        #
        #     source = self.experience
        #
        #     if prioritized:
        #         source = self.experience_prioritized
        #         logger.debug('Sampling prioritized only, %s from %s', batch_size, len(source))
        #
        #     if len(source) < 1:
        #         logger.warning('No samples for training available.')
        #         return
        #
        #     for x in range(episodes):
        #
        #         batches = np.random.choice(len(source), batch_size)
        #
        #         states = [None] * batch_size
        #         actions = [None] * batch_size
        #         targets = np.zeros((batch_size, 1))
        #
        #         for i in range(batch_size):
        #             state, action, reward, state_next, actions_next, finished = source[batches[i]]
        #             target = reward
        #
        #             if not finished:
        #                 # get an action with maximum Q value
        #                 q_max = -np.math.inf
        #                 for action_next in actions_next:
        #                     q = self.q(state_next, action_next)
        #                     if q > q_max:
        #                         q_max = q
        #                 target += gamma * q_max
        #
        #             states[i] = state
        #             actions[i] = action
        #             targets[i] = target
        #
        #         states = pad_sequences(states)
        #         actions = pad_sequences(actions)
        #
        #         self.model.fit(x=[states, actions], y=targets, batch_size=batch_size, epochs=1, verbose=1)
