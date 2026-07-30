from abc import ABC, abstractmethod
import random

class Scheduler(ABC):

    def __init__(self, name : str, full_name : str, seed = 123):
        self.server_farm = None
        self.servers = None
        self.name = name
        self.full_name = full_name
        self.rng = random.Random(seed)

    @abstractmethod
    def assign_tasks(self):
        pass