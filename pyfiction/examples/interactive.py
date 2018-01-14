"""
This program enables to user to play any of the supported games interactively.
"""
from pyfiction.simulators.games.catsimulator2016_simulator import CatSimulator2016Simulator
from pyfiction.simulators.games.howlingdogs_simulator import HowlingDogsSimulator
from pyfiction.simulators.games.machineofdeath_simulator import MachineOfDeathSimulator
from pyfiction.simulators.games.savingjohn_simulator import SavingJohnSimulator
from pyfiction.simulators.games.starcourt_simulator import StarCourtSimulator
from pyfiction.simulators.games.theredhair_simulator import TheRedHairSimulator
from pyfiction.simulators.games.transit_simulator import TransitSimulator
from pyfiction.simulators.text_games.simulators.MySimulator import StoryNode

simulators = [SavingJohnSimulator, MachineOfDeathSimulator, CatSimulator2016Simulator, StarCourtSimulator,
              TheRedHairSimulator, TransitSimulator, HowlingDogsSimulator]


def select_game():
    print("Play an interactive fiction game supported by pyfiction.")
    print()
    print("When selecting games or actions, type their index to pick one of them"
          " or type 'exit' to end the program or the current game.")
    print()
    print("Available game simulators:")
    for i, simulator in enumerate(simulators):
        print('[' + str(i) + '] ' + simulator.__name__)

    exit_game = False
    while True:
        try:
            index = input('Choose a game simulator by its index (e.g. for ' + simulators[0].__name__ + " type '0'): ")
            if index.lower() == 'exit':
                exit_game = True
                break
            simulator = simulators[int(index)]
            break
        except Exception as e:
            print('Error:', e)

    if exit_game:
        return 0

    play_game(simulator)


def play_game(simulator):
    simulator = simulator(shuffle_actions=False)
    simulator.restart()

    print()
    print('Playing game ' + simulator.game.name)

    cumulative_reward = 0

    while True:
        print('----------------------------------------------------------------')
        state, actions, reward = simulator.read()
        print(state)
        print('--------------------------------')
        print('Reward:', reward)
        print('Actions:')

        cumulative_reward += reward

        if not actions:
            print('----------------------------------------------------------------')
            print('Game ended! Cumulative reward: ' + str(cumulative_reward))
            print('Going back to game selection menu.')
            print('----------------------------------------------------------------')
            break

        for i, action in enumerate(actions):
            print('[' + str(i) + '] ' + action)

        print('--------------------------------')

        exit_game = False
        while True:
            try:
                action_input = input('Choose an action: ')
                if action_input.lower() == 'exit':
                    exit_game = True
                    break
                simulator.write(int(action_input))
                break
            except Exception as e:
                print('Error:', e)
        if exit_game:
            print('----------------------------------------------------------------')
            print('Exiting game! Cumulative reward: ' + str(cumulative_reward))
            print('Going back to game selection menu.')
            print('----------------------------------------------------------------')
            break

    simulator.close()
    select_game()


select_game()
