from selenium import webdriver

from pyfiction.simulators.simulator import Simulator


class HTMLSimulator(Simulator):
    def __init__(self, game, shuffle_actions=True):
        self.game = game
        self.shuffle_actions = shuffle_actions

        # use a headless Chromium/Chrome browser
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("headless")
        chrome_options.add_argument("disable-gpu")
        self.driver = webdriver.Chrome(chrome_options=chrome_options)

        self.driver.get('file:///' + self.game.path)

    def restart(self):
        self.driver.get('file:///' + self.game.path)

    def startup_actions(self):
        for action in self.game.startup_actions:
            self.read()
            self.write(action)

    def write(self, text):
        pass

    def read(self):
        pass
