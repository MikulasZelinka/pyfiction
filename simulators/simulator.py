from subprocess import *
from simulators.nbstreamreader import NonBlockingStreamReader


class Simulator:
    def __init__(self, game):
        self.__game = game
        self.__game_type = None
        self.__path = None
        self.__interpreter_path = None
        self.__nbsr = None
        self.simulator = None

    def start_game(self, interpreter_path=None, game_path=None):

        self.simulator = Popen([self.__game.interpreter.path, self.__game.path], stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                               bufsize=0, universal_newlines=False)
        print('Running interpreter ', self.__game.interpreter.path, ' on game ', self.__game.path)

        self.__nbsr = NonBlockingStreamReader(self.simulator.stdout)
        print('Game started')

    def startup_actions(self):
        for action in self.__game.startup_actions:
            print(self.read())
            self.write(action)
        print('--------------------------------------')
        print('all startup actions have been executed')
        print('--------------------------------------')

    def write(self, text):
        # print('writing->', repr(text), '<-')
        self.simulator.stdin.flush()
        self.simulator.stdin.write(text.encode('utf-8'))
        # print('write end')

    # TODO: add a regexp 'timeout' (e.g. read until '\n >' is read)
    def read(self, timeout=0.01):
        # print('read start')
        lines = []
        while True:
            line = self.__nbsr.read_line(timeout)
            if line is not None:
                if not (line == '\n'):
                    lines.append(line)
                # print(output)
                self.simulator.stdout.flush()
            else:
                # print('[No more data]')
                break
                # print(output)

        # print('read end')
        return lines
