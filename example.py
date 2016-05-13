from game import Game
import time

game = Game(path='')
game.start_game()


i = 0
while True:
    time.sleep(1)

    if i == 0:
        game.write('cd ..\n')
    elif i == 1:
        game.write('ls\n')
    elif i == 2:
        game.write('cd ..\n')
    elif i == 3:
        game.write('ls -l\n')

    print(game.read())
    i += 1
