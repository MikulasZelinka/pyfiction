import logging

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

agent = LSTMAgent(simulator=SavingJohnSimulator)
agent.sample(episodes=8192)
agent.create_model()

for i in range(256):
    logger.info('Epoch %s', i)
    agent.train(batch_size=32, prioritised=i < 4)
    reward = agent.test(iterations=1)
    logger.info('Average reward: %s', reward)
