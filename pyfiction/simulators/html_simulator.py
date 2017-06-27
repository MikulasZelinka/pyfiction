from selenium import webdriver

from pyfiction.simulators.simulator import Simulator


class HTMLSimulator(Simulator):

    def __init__(self, game, shuffle=True):
        self.driver = webdriver.Chrome()
        self.game = game
        self.driver.get('file:///' + self.game.path)
        self.shuffle = shuffle

    def restart(self):
        pass

    def startup_actions(self):
        for action in self.game.startup_actions:
            self.read()
            self.write(action)

    def write(self, text):
        pass

    def read(self):
        pass

