import logging

from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import plot_model
from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.games.machineofdeath_simulator import MachineOfDeathSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example agent for Machine of Death that tests transfer learning on paraphrased actions
"""

simulator = MachineOfDeathSimulator(paraphrase_actions=False)
simulator_paraphrased = MachineOfDeathSimulator(paraphrase_actions=True)
# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(train_simulators=simulator_paraphrased, test_simulators=[simulator_paraphrased, simulator])

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

agent.model = load_model('logs/ep8191_07-10-02_21_40')

# Visualize the model
try:
    plot_model(agent.model, to_file='model.png', show_shapes=True)
except ImportError as e:
    logger.warning("Couldn't print the model image: {}".format(e))

# Iteratively train the agent on a batch of previously seen examples while continuously expanding the experience buffer
# This example seems to converge to nearly optimal rewards in all three game branches
epochs = 1
for i in range(epochs):
    logger.info('Epoch %s', i)
    agent.train_online(episodes=8192, batch_size=256, gamma=0.95, epsilon_decay=0.99,
                       prioritized_fraction=0.25, test_interval=8, test_steps=5)

