import logging

from keras.optimizers import RMSprop
from keras.utils import plot_model
from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.games.theredhair_simulator import TheRedHairSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example agent for The Red Hair that uses online learning and prioritized sampling
"""

# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(simulator=TheRedHairSimulator)

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens(iterations=2 ** 5, max_steps=100)

optimizer = RMSprop()

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
epochs = 1
for i in range(epochs):
    logger.info('Epoch %s', i)
    agent.train_online(episodes=1024, batch_size=256, gamma=0.2, epsilon=1,
                       epsilon_decay=0.999, prioritized_fraction=0.25, test_interval=4)
agent.simulator.close()
