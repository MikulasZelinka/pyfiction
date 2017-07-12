import argparse
import logging

from keras.models import load_model
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
An LSTM agent that supports leave-one-out generalisation testing
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
                    help='index of a simulator to use for leave-one-out testing [1-6]',
                    type=int,
                    default=0)
parser.add_argument('--model',
                    help='file path of a model to load',
                    type=str)

args = parser.parse_args()
simulator_index = args.simulator
model_path = args.model

simulator = simulators[simulator_index - 1]
test_steps = test_steps[simulator_index - 1]
print('Training on game:', simulator.game.name)
print('Testing on game:', simulator.game.name)

agent = LSTMAgent(train_simulators=simulator)

# Load or learn the vocabulary (random sampling on many games could be extremely slow)
agent.initialize_tokens('vocabulary.txt')

optimizer = RMSprop(lr=0.00001)

embedding_dimensions = 16
lstm_dimensions = 32
dense_dimensions = 8

agent.create_model(embedding_dimensions=embedding_dimensions,
                   lstm_dimensions=lstm_dimensions,
                   dense_dimensions=dense_dimensions,
                   optimizer=optimizer)

agent.model = load_model(model_path)

# Visualize the model
try:
    plot_model(agent.model, to_file='model.png', show_shapes=True)
except ImportError as e:
    logger.warning("Couldn't print the model image: {}".format(e))

# Transfer learning test - train the agent on the previously unseen (only used for testing) game

agent.train_online(episodes=8192, batch_size=256, gamma=0.95, epsilon=1, epsilon_decay=0.999,
                   prioritized_fraction=0.25, test_interval=16, test_steps=test_steps,
                   log_prefix=('transfer' + str(simulator_index)))
