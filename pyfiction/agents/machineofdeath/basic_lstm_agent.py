import logging

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.machineofdeath_simulator import MachineOfDeathSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

agent = LSTMAgent(simulator=MachineOfDeathSimulator, state_length=256, action_length=16)
agent.sample(episodes=8192)
agent.create_model()

for i in range(256):
    logger.info('Epoch %s', i)
    agent.train(batch_size=64, prioritised=i < 64)
    reward = agent.test(iterations=10)
    logger.info('Average reward: %s', reward)
