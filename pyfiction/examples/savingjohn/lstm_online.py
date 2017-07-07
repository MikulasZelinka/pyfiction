import logging

from keras.optimizers import RMSprop
from keras.utils import plot_model
from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example agent for Saving John that uses online learning and prioritized sampling
"""

# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(train_simulators=SavingJohnSimulator())

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens()

optimizer = RMSprop(lr=0.001)

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
# This example seems to converge to a reward of 19.3+ (with 19.4 being the optimal reward)
epochs = 1
for i in range(epochs):
    logger.info('Epoch %s', i)
    agent.train_online(episodes=128, batch_size=256, gamma=0.95, epsilon=1, epsilon_decay=0.99,
                       prioritized_fraction=0.25, test_interval=1, test_steps=1)

    # inspect model weights:
    for layer in agent.model.layers:
        print('layer', layer.name)
        print(layer.get_weights())
