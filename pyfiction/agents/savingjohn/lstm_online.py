import logging

from keras.optimizers import SGD, RMSprop

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
agent = LSTMAgent(simulator=SavingJohnSimulator, state_length=64, action_length=12)

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens(iterations=1024, max_steps=100)

# Create a model with given parameters
optimizer = RMSprop(lr=0.00005)
embedding_dimensions = 16
agent.create_model(embedding_dimensions=embedding_dimensions,
                   dense_dimensions=8,
                   optimizer=optimizer)

# Iteratively train the agent on a batch of previously seen examples while continuously expanding the experience buffer
# Somehow consider prioritized experiences
epochs = 256
for i in range(epochs):
    logger.info('Epoch %s', i)
    agent.train_online(episodes=1, max_steps=100, batch_size=32)

    # Only one run of the game is necessary for testing since both the agent (epsilon=0) and the game are deterministic
    reward = agent.play_game(episodes=1, store_experience=False, max_steps=100, epsilon=0, verbose=False)
    logger.info('Average reward: %s', reward)
