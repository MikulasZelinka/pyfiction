import logging

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.machineofdeath_simulator import MachineOfDeathSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

agent = LSTMAgent(simulator=MachineOfDeathSimulator)
agent.sample(episodes=8192)
agent.create_model()

for i in range(256):
    logger.info('Epoch %s', i)
    agent.train(batch_size=32, prioritised=i < 32)
    reward = agent.test(iterations=10)
    logger.info('Average reward: %s', reward)
