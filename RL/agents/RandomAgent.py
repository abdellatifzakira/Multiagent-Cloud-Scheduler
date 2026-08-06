try:
    from Agent import Agent
except ModuleNotFoundError:
    from RL.agents.Agent import Agent

class RandomAgent(Agent):
    def __init__(self, seed = 123):
        super().__init__('RA', 'RANDOM AGENT')
        self.trainable = False
    
    def take_action(self):
        action = self.action_space.sample()
        return action
    
    
    def observe(self, *state):
        pass

    def build(self):
        pass
    
    def update(self, *args):
        pass
    