import logging

from keras.optimizers import SGD
from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
THIS EXAMPLE IS CURRENTLY OBSOLETE (breaking changes to the LSTMAgent class were introduces)
An example agent for Saving John that uses offline learning and prioritized sampling
The samples are obtained using a random policy, meaning this method is only suitable for games where a random policy
 samples the states evenly
This agent class is universal and it should be possible to apply it to different games in the same way
"""

# Create the agent and specify maximum lengths of descriptions (in words)
agent = LSTMAgent(simulator=SavingJohnSimulator, state_length=64, action_length=12)

# Learn the vocabulary (the function samples the game using a random policy)
agent.initialize_tokens(iterations=1024, max_steps=100)

# Create a model with given parameters, also using pre-trained GloVe embeddings (not necessary):
optimizer = SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
# optimizer = RMSprop(lr=0.0005, clipvalue=0.5)
# TODO - add regularization to avoid inf/nans on loss function (and possibly on weights?)
embedding_dimensions = 50
embeddings = 'glove.6B.' + str(embedding_dimensions) + 'd.txt'
# To train embeddings from scratch, simply delete the 'embeddings' parameter from the function call
agent.create_model(embedding_dimensions=embedding_dimensions,
                   dense_dimensions=8,
                   optimizer=optimizer,
                   embeddings=embeddings,
                   embeddings_trainable=False)  # only applies if embeddings parameter is present

# Let the agent play the game using a random policy (ensured by epsilon=1) while storing the experiences:
agent.play_game(episodes=1024, max_steps=100, epsilon=1, store_experience=True)

# Now iteratively train the agent on a batch of previously randomly sampled examples and test after each iteration
# First i < X batches will only learn from prioritized experiences (i.e. those of final states with large rewards)
epochs = 256
for i in range(epochs):
    logger.info('Epoch %s', i)
    agent.train_offline(episodes=1, batch_size=1, prioritized=i < 16)

    # Only one run of the game is necessary for testing since both the agent (epsilon=0) and the game are deterministic
    # Note that store_experience is set to False and the agent only learns from the experiences sampled before learning
    reward = agent.play_game(episodes=1, store_experience=False, max_steps=100, epsilon=0, verbose=False)
    logger.info('Average reward: %s', reward)
