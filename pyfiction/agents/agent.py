class Agent(object):
    def __init__(self):
        raise NotImplementedError("Agent is an abstract class.")

    def act(self, observation):
        raise NotImplementedError("Agent is an abstract class.")
