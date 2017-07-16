import argparse
import logging

from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import plot_model
from pyfiction.agents.lstm_agent import LSTMAgent
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Load a model of an agent and interactively test its Q-values of the state and action texts supplied by the user
"""

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    help='file path of a model to load',
                    type=str,
                    default='all.h5')
# all.h5 contains a model trained on all six games (generalisation.py with argument of 0)

args = parser.parse_args()
model_path = args.model

agent = LSTMAgent(None)

# Load or learn the vocabulary (random sampling on many games could be extremely slow)
agent.initialize_tokens('vocabulary.txt')

optimizer = RMSprop(lr=0.00001)

embedding_dimensions = 16
lstm_dimensions = 32
dense_dimensions = 8

agent.create_model(embedding_dimensions=embedding_dimensions,
                   lstm_dimensions=lstm_dimensions,
                   dense_dimensions=dense_dimensions,
                   optimizer=optimizer)

agent.model = load_model(model_path)

print("Model", model_path, "loaded, now accepting state and actions texts and evaluating their Q-values.")

while True:
    state = input("State: ")
    action = input("Action: ")
    print("Q-value: ", agent.q(state, action) * 30)
    print("------------------------------")
