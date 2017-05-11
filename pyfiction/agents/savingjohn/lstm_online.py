import logging

from keras.optimizers import SGD

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

optimizer = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)

agent = LSTMAgent(simulator=SavingJohnSimulator, optimizer=optimizer, max_steps=100)

epochs = 256

for i in range(epochs):
    logger.info('Epoch %s', i)
    agent.train_online(episodes=1024, batch_size=32, gamma=0.99, epsilon=1, epsilon_decay=0.99, prioritised=False)
