import logging

from keras.optimizers import RMSprop
from keras.utils import plot_model
from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.games.catsimulator2016_simulator import CatSimulator2016Simulator
from pyfiction.simulators.games.machineofdeath_simulator import MachineOfDeathSimulator
from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.games.starcourt_simulator import StarCourtSimulator
from pyfiction.simulators.games.theredhair_simulator import TheRedHairSimulator
from pyfiction.simulators.games.transit_simulator import TransitSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example agent for Machine of Death that uses online learning and prioritized sampling
"""

simulators = [CatSimulator2016Simulator(),
              MachineOfDeathSimulator(),
              SavingJohnSimulator(),
              # StarCourtSimulator(),
              TheRedHairSimulator(),
              TransitSimulator()]

# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(train_simulators=simulators)

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
# This example seems to converge to nearly optimal rewards in all three game branches
epochs = 0
for i in range(epochs):
    logger.info('Epoch %s', i)
    agent.train_online(episodes=256 * 256, batch_size=256, gamma=0.95, epsilon=1, epsilon_decay=0.999,
                       prioritized_fraction=0.25, test_interval=16, test_steps=5)