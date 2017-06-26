import logging

from keras.optimizers import RMSprop
from keras.utils import plot_model
from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.games.machineofdeath_simulator import MachineOfDeathSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example agent for Machine of Death that uses online learning and prioritized sampling
"""

# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(simulator=MachineOfDeathSimulator)

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens(iterations=8192, max_steps=500)

optimizer = RMSprop(lr=0.0005)

embedding_dimensions = 16
lstm_dimensions = 16
dense_dimensions = 4

agent.create_model(embedding_dimensions=embedding_dimensions,
                   lstm_dimensions=lstm_dimensions,
                   dense_dimensions=dense_dimensions,
                   optimizer=optimizer)

# Visualize the model
try:
    plot_model(agent.model, to_file='model.png', show_shapes=True)
except ImportError as e:
    logger.warning("Couldn't print the model image: {}".format(e))

# Iteratively train the agent on a batch of previously seen examples while continuously expanding the experience buffer
# This example seems to converge to nearly optimal rewards in two out of three game branches
epochs = 1

for i in range(epochs):
    logger.info('Epoch %s', i)
    rewards = agent.train_online(episodes=256*256, max_steps=500, batch_size=256, gamma=0.95, epsilon=1,
                                 epsilon_decay=0.99, reward_scale=30, prioritized_fraction=0.25, test_steps=4)

# Test on paraphrased actions
# agent.simulator = MachineOfDeathSimulator(paraphrase_actions=True)
# agent.model = load_model('logs/Epoch3_06-14-05_29_25.h5')
# episodes = 256
# for i in range(episodes):
#     logger.info('Final reward: %s',
#                 agent.play_game(max_steps=500, episodes=8, verbose=False, store_experience=False, epsilon=0))

# for i in range(episodes):
#     logger.info('Train reward: %s',
#                 agent.train_online(episodes=1, max_steps=500, batch_size=128, gamma=0.95, epsilon=0, reward_scale=20,
#                                    epsilon_decay=0, prioritized_fraction=0.25, test_steps=1))
