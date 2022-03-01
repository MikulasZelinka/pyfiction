import argparse
import logging
import string

import tensorflow as tf
from keras.utils.vis_utils import plot_model
from pyfiction.agents.ssaqn_agent import SSAQNAgent
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
An SSAQN agent that supports leave-one-out generalisation testing
"""

simulators = [CatSimulator2016Simulator(),
              MachineOfDeathSimulator(),
              SavingJohnSimulator(),
              StarCourtSimulator(),
              TheRedHairSimulator(),
              TransitSimulator()]
test_steps = [
    1,
    5,
    1,
    5,
    1,
    1
]

parser = argparse.ArgumentParser()
parser.add_argument('--simulator',
                    help='index of a simulator to use for leave-one-out testing [1-6], 0 for training and testing all',
                    type=int,
                    default=0)

parser.add_argument('--log_folder',
                    help='a folder to store logs in, default is "logs"',
                    type=str,
                    default="logs")

args = parser.parse_args()
simulator_index = args.simulator
log_folder = args.log_folder

if simulator_index == 0:
    train_simulators = simulators
    test_simulators = simulators
    print('Training and testing on all games:', [simulator.game.name for simulator in simulators])
else:
    train_simulators = simulators[:simulator_index - 1] + simulators[simulator_index:]
    test_simulators = simulators[simulator_index - 1]
    test_steps = test_steps[simulator_index - 1]
    print('Training on games:', [simulator.game.name for simulator in train_simulators])
    print('Testing on game:', test_simulators.game.name)

# Create the agent and specify maximum lengths of descriptions (in words)
agent = SSAQNAgent(train_simulators=train_simulators, test_simulators=test_simulators, log_folder=log_folder)

# Load or learn the vocabulary (random sampling on this many games could be extremely slow)
agent.initialize_tokens('vocabulary.txt')

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001)

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

# Iteratively train the agent on five out of the six games or on all six games
# This example seems to converge to the optimal reward in all games but Star Court  when training on all games
epochs = 1
for i in range(epochs):
    logger.info('Epoch %s', i)
    agent.train_online(episodes=8192, batch_size=256, gamma=0.95, epsilon=1, epsilon_decay=0.999,
                       prioritized_fraction=0.25, test_interval=16, test_steps=test_steps,
                       log_prefix=str(simulator_index))

# Transfer learning test - train the agent on the previously unseen (only used for testing) game
if simulator_index != 0:
    agent.clear_experience()
    agent.train_simulators = test_simulators if isinstance(test_simulators, list) else [test_simulators]
    agent.train_online(episodes=8192, batch_size=256, gamma=0.95, epsilon=1, epsilon_decay=0.999,
                       prioritized_fraction=0.25, test_interval=16, test_steps=test_steps,
                       log_prefix=('transfer' + str(simulator_index)))
