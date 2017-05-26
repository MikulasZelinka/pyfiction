import datetime
import logging

from keras.optimizers import RMSprop
from keras.utils import plot_model

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.machineofdeath_simulator import MachineOfDeathSimulator

# from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example agent for Machine of Death that uses online learning and prioritized sampling
This agent class is universal and it should be possible to apply it to different games in the same way
"""

# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(simulator=MachineOfDeathSimulator, state_length=256, action_length=16)

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens(iterations=8092, max_steps=500)

# Create a model with given parameters
optimizer = RMSprop()  # (lr=0.0005)
# optimizer = SGD(lr=0.001)
embedding_dimensions = 16
agent.create_model(embedding_dimensions=embedding_dimensions,
                   dense_dimensions=8,
                   optimizer=optimizer)

# Visualize the model
plot_model(agent.model, to_file='model.png', show_shapes=True)

# Iteratively train the agent on a batch of previously seen examples while continuously expanding the experience buffer
# This example seems to converge to a reward of 19.93 (with 19.94 being the optimal reward)
epochs = 128
for i in range(epochs):
    logger.info('Epoch %s', i)
    rewards = agent.train_online(episodes=1024, max_steps=500, batch_size=64, gamma=0.95, epsilon=1,
                                 epsilon_decay=0.995, prioritized_fraction=0.25)
    file_name = 'Epoch' + str(i) + '_' + str(datetime.datetime.now())
    with open(file_name + '.txt', 'w') as file:
        for reward in rewards:
            file.write(str(reward) + '\n')
    agent.model.save(file_name + '.h5')
