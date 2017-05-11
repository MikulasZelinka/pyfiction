import logging

from keras.optimizers import SGD

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.machineofdeath_simulator import MachineOfDeathSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

optimizer = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)

agent = LSTMAgent(simulator=MachineOfDeathSimulator, state_length=256, action_length=16, optimizer=optimizer,
                  max_steps=64)
agent.sample(episodes=8192)
agent.create_model()

for i in range(1024):
    logger.info('Epoch %s', i)
    agent.train(batch_size=32, prioritised=i < 64)
    reward = agent.test(iterations=10, epsilon=0.05, verbose=False)
    logger.info('Average reward: %s', reward)
