from simulator.simulator import Simulator
from games.Six.six import Six
import time

simulator = Simulator(Six)
simulator.start_game()


i = 0
while True:
    time.sleep(1)

    if i == 0:
        simulator.write('cd ..\n')
    elif i == 1:
        simulator.write('ls\n')
    elif i == 2:
        simulator.write('cd ..\n')
    elif i == 3:
        simulator.write('ls -l\n')

    print(simulator.read())
    i += 1
