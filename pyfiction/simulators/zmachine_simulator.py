from pyfiction.simulators.simulator import Simulator


class ZMachineSimulator(Simulator):
    def __init__(self):
        raise NotImplementedError

    def restart(self, interpreter_path=None, game_path=None):
        raise NotImplementedError

    def __startup_actions(self):
        raise NotImplementedError

    def read(self, timeout=0.01):
        raise NotImplementedError

    def write(self, text):
        raise NotImplementedError
