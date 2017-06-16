import datetime
import logging
import os

from keras.models import load_model
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

optimizer = RMSprop(lr=0.0005)

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
    print('Couldn\'t print the model image: ', e)

# Iteratively train the agent on a batch of previously seen examples while continuously expanding the experience buffer
# This example seems to converge to two nearly optimal rewards in two out of three game branches
epochs = 16
os.makedirs('logs', exist_ok=True)
for i in range(epochs):
    logger.info('Epoch %s', i)
    rewards = agent.train_online(episodes=1024, max_steps=500, batch_size=256, gamma=0.95, epsilon=1, reward_scale=30,
                                 epsilon_decay=0.99, prioritized_fraction=0.25, test_steps=4)
    # rewards = agent.train_traces(episodes=1024, max_steps=500, batch_size=512, gamma=0.99, epsilon_decay=0.995,
    #                              test_steps=4, reward_scale=20)
    file_name = 'Epoch' + str(i) + '_' + datetime.datetime.now().strftime('%m-%d-%H_%M_%S')
    with open('logs/' + file_name + '.txt', 'w') as file:
        for reward in rewards:
            file.write(str(reward) + '\n')
    agent.model.save('logs/' + file_name + '.h5')

# Test on paraphrased actions
# agent.simulator = MachineOfDeathSimulator(paraphrase_actions=True)
# agent.model = load_model('logs/Epoch3_06-14-05_29_25.h5')
# episodes = 256
# for i in range(episodes):
#     logger.info('Final reward: %s',
#                 agent.play_game(max_steps=500, episodes=1, verbose=False, store_experience=False, epsilon=0))

# for i in range(episodes):
#     logger.info('Train reward: %s',
#                 agent.train_online(episodes=1, max_steps=500, batch_size=128, gamma=0.95, epsilon=0, reward_scale=20,
#                                    epsilon_decay=0, prioritized_fraction=0.25, test_steps=1))
