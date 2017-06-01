import datetime
import logging

from keras.optimizers import RMSprop
from keras.utils import plot_model

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.machineofdeath_simulator import MachineOfDeathSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example agent for Machine of Death that uses online learning and prioritized sampling
This agent class is universal and it should be possible to apply it to different games in the same way
"""

# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(simulator=MachineOfDeathSimulator)

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens(iterations=8092, max_steps=500)

# Create a model with given parameters
optimizer = RMSprop()  # (lr=0.0005)
# optimizer = SGD(lr=0.001)
embedding_dimensions = 32
lstm_dimensions = 16
dense_dimensions = 8
agent.create_model(embedding_dimensions=embedding_dimensions,
                   lstm_dimensions=lstm_dimensions,
                   dense_dimensions=dense_dimensions,
                   optimizer=optimizer)

# Visualize the model
plot_model(agent.model, to_file='model.png', show_shapes=True)

# Iteratively train the agent on a batch of previously seen examples while continuously expanding the experience buffer
epochs = 128
for i in range(epochs):
    logger.info('Epoch %s', i)
    rewards = agent.train_online(episodes=128, max_steps=500, batch_size=64, gamma=0.95, epsilon=1,
                                 epsilon_decay=0.99, prioritized_fraction=0.25, test_steps=12)
    file_name = 'Epoch' + str(i) + '_' + str(datetime.datetime.now())
    with open(file_name + '.txt', 'w') as file:
        for reward in rewards:
            file.write(str(reward) + '\n')
    agent.model.save(file_name + '.h5')
