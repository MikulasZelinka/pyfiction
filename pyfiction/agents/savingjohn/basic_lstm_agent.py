import logging

from keras.optimizers import SGD, RMSprop

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# converges to 19.93 reward (optimal is 19.94):
optimizer = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)

# optimizer = RMSprop()

agent = LSTMAgent(simulator=SavingJohnSimulator, optimizer=optimizer)
agent.sample(episodes=8192)
agent.create_model()

for i in range(256):
    logger.info('Epoch %s', i)
    agent.train(batch_size=32, prioritised=i < 32)
    reward = agent.test(iterations=1, verbose=True)
    logger.info('Average reward: %s', reward)
