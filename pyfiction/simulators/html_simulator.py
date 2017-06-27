from selenium import webdriver

from pyfiction.simulators.simulator import Simulator


class HTMLSimulator(Simulator):

    def __init__(self, game, shuffle=True):
        self.driver = webdriver.Chrome()
        self.game = game
        self.driver.get('file:///' + self.game.path)
        self.shuffle = shuffle
        # self.restart()

    def restart(self):
        pass

    def __startup_actions(self):
        # for action in self.game.__startup_actions:
        #     self.read()
        #     self.write(action)
        pass

    def write(self, text):
        pass

    def read(self):
        pass

