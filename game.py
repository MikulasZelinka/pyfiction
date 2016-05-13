from subprocess import *
from supported_games import GameType
from nbstreamreader import NonBlockingStreamReader
import os


class Game:
    @property
    def game(self):
        return self.__game

    @game.setter
    def game(self, game):
        self.__game = game

    @property
    def game_type(self):
        return self.__game_type

    @game_type.setter
    def game_type(self, game_type):
        self.__game_type = game_type

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        self.__path = path

    @property
    def interpreter_path(self):
        return self.__interpreter_path

    @interpreter_path.setter
    def interpreter_path(self, interpreter_path):
        self.__interpreter_path = interpreter_path

    def __init__(self, path, game_type=GameType.Default):
        self.__game = None
        self.__game_type = None
        self.__path = None
        self.__interpreter_path = None
        self.__nbsr = None
        self.path = path
        self.game_type = game_type

    def start_game(self):
        interpreter_path = "interpreters/cheapglulxe"
        if os.name != "posix":
            interpreter_path += ".exe"

        # self.game = Popen([self.interpreter_path, self.path], stdin=PIPE, stdout=PIPE, stderr=STDOUT,
        #                     bufsize=1, universal_newlines=True)

        # using cmd for testing purposes insted of an IF interpreter + game
        self.game = Popen(['cmd'], stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                          bufsize=1, universal_newlines=True)

        self.__nbsr = NonBlockingStreamReader(self.game.stdout)
        print('game started')

    def write(self, text):
        print('write start')
        self.game.stdin.flush()
        self.game.stdin.write(text)
        print('write end')

    # TODO: add a regexp 'timeout' (e.g. read until '\n >' is read)
    def read(self, timeout=0.001):
        print('read start')
        lines = []
        while True:
            line = self.__nbsr.readline(timeout)
            if line is not None:
                lines.append(line)
                # print(output)
                self.game.stdout.flush()
            else:
                # print('[No more data]')
                break
                # print(output)

        print('read end')
        return lines
