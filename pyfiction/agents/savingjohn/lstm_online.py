import datetime
import logging
import os

from keras.optimizers import SGD, RMSprop
from keras.utils import plot_model

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
An example agent for Saving John that uses online learning and prioritized sampling
This agent class is universal and it should be possible to apply it to different games in the same way
"""

# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(simulator=SavingJohnSimulator)

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens(iterations=1024, max_steps=100)

optimizer = RMSprop()

# embedding_dimensions = 32
# lstm_dimensions = 16
# dense_dimensions = 8

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
    print('Couldn\'t print the model image: ', e)

# Iteratively train the agent on a batch of previously seen examples while continuously expanding the experience buffer
# This example seems to converge to a reward of 19.9X (with 19.94 being the optimal reward)
epochs = 1
os.makedirs('logs', exist_ok=True)
for i in range(epochs):
    logger.info('Epoch %s', i)
    rewards = agent.train_online(episodes=1024, max_steps=100, batch_size=64, gamma=0.95, epsilon=1, reward_scale=20,
                                 epsilon_decay=0.99, prioritized_fraction=0.25, test_steps=4)
    # rewards = agent.train_traces(episodes=1024, max_steps=100, batch_size=64, gamma=0.95, epsilon_decay=0.995,
    #                              test_steps=1, reward_scale=20)
    file_name = 'Epoch' + str(i) + '_' + datetime.datetime.now().strftime('%m-%d-%H_%M_%S')
    with open('logs/' + file_name + '.txt', 'w') as file:
        for reward in rewards:
            file.write(str(reward) + '\n')
    agent.model.save('logs/' + file_name + '.h5')
