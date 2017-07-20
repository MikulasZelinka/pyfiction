import logging

from keras.optimizers import RMSprop
from keras.utils import plot_model
from pyfiction.agents.ssaqn_agent import SSAQNAgent
from pyfiction.simulators.games.starcourt_simulator import StarCourtSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example SSAQN agent for Star Court that uses online learning and prioritized sampling
"""

# Create the agent and specify maximum lengths of descriptions (in words)
agent = SSAQNAgent(train_simulators=StarCourtSimulator())

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens('vocabulary.txt')

optimizer = RMSprop(lr=0.00001)

embedding_dimensions = 16
lstm_dimensions = 32
dense_dimensions = 8

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
# This example seems to not converge but we do not know if there exists a policy reaching consistently good rewards
epochs = 1
for i in range(epochs):
    logger.info('Epoch %s', i)
    agent.train_online(episodes=256 * 256, batch_size=256, gamma=0.95, epsilon_decay=0.999,
                       prioritized_fraction=0.25, test_interval=8, test_steps=5)
