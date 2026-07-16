import numpy as np

class RoundRobinScheduler:
    def __init__(self,
                 ):
        self.server_farm = None
        self.servers = None
        self.pointer = 0
    
        
    def assign_tasks(self, ready_tasks, t):
        n = len(self.servers)
        for task in ready_tasks :
                    server = self.servers[self.pointer]
                    success = server.add_task_to_queue(task, t)
                    if success :
                        self.pointer = (self.pointer + 1) % n


