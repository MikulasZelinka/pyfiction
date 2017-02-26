from subprocess import *

from pyfiction.interpreters.glulx.glulxe import Glulxe
from pyfiction.simulators.nbstreamreader import NonBlockingStreamReader
from pyfiction.simulators.simulator import Simulator


class GlulxSimulator(Simulator):
    interpreter = Glulxe

    def __init__(self, game):
        self.game = game
        self.stream = None
        self.stream_reader = None

    def restart(self):

        print('Running interpreter ', self.interpreter.path, ' on game ', self.game.path)
        self.stream = Popen([self.interpreter.path, self.game.path], stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                            bufsize=0, universal_newlines=False)

        self.stream_reader = NonBlockingStreamReader(self.stream.stdout)

        self.__startup_actions()
        print('Game started')

    def __startup_actions(self):
        for action in self.game.__startup_actions:
            # print(self.read())
            self.read()
            self.write(action)
            # print('--------------------------------------')
            # print('all startup actions have been executed')
            # print('--------------------------------------')

    def write(self, text):
        # print('writing->', repr(text), '<-')
        self.stream.stdin.flush()
        self.stream.stdin.write(text.encode('utf-8'))
        # print('write end')

    # TODO: add a regexp 'timeout' (e.g. read until '\n >' is read)
    def read(self, timeout=0.01):
        # print('read start')
        lines = []
        while True:
            line = self.stream_reader.read_line(timeout)
            if line is not None:
                if not (line == '\n'):
                    lines.append(line)
                # print(output)
                self.stream.stdout.flush()
            else:
                # print('[No more data]')
                break
                # print(output)

        # print('read end')
        return lines
