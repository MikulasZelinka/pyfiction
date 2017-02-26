class Agent(object):
    def __init__(self):
        raise NotImplementedError("Agent is an abstract class.")

    def act(self, **kwargs):
        raise NotImplementedError("Agent is an abstract class.")
