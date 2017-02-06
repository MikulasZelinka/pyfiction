from pyfiction.games.Six.six import Six
from pyfiction.simulators.glulx_simulator import GlulxSimulator


class SixSimulator(GlulxSimulator):
    def __init__(self):
        super().__init__(Six)
