import logging

from keras.optimizers import SGD

from pyfiction.agents.lstm_agent import LSTMAgent
from pyfiction.simulators.machineofdeath_simulator import MachineOfDeathSimulator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

optimizer = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)

agent = LSTMAgent(simulator=MachineOfDeathSimulator, state_length=256, action_length=16, optimizer=optimizer,
                  max_steps=256)
agent.create_model()

epsilon = 1
for i in range(1024):

    logger.info('Epoch %s', i)

    reward = agent.play_game(episodes=1, store_experience=True, epsilon=epsilon, verbose=True)
    logger.info('Average reward: %s', reward)

    agent.train_offline(batch_size=32, prioritised=False)

    epsilon *= 0.99
