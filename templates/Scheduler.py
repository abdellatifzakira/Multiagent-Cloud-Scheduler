from abc import ABC, abstractmethod

class Scheduler(ABC):

    def __init__(self, name : str):
        self.server_farm = None
        self.servers = None
        self.name = name

    @abstractmethod
    def assign_tasks(self):
        pass