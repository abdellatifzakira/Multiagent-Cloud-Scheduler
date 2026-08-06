from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, name, full_name):
        self.action_space = None
        self.name = name
        self.full_name = full_name
        self.trainable = True
    
    @abstractmethod
    def take_action(self):
        raise NotImplementedError("Take action method not implemented !")
    
    
    @abstractmethod
    def observe(self):
        raise NotImplementedError("Observe method not implemented !")

    @abstractmethod
    def build(self):
        raise NotImplementedError("build method not implemented !")
    

    @abstractmethod
    def update(self, reward, done):
        raise NotImplementedError("Update method not implemented !")