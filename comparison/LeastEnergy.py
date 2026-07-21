import numpy as np
import random
import math


class EnergyAwareScheduler:

    def __init__(self):
        self.server_farm = None
        self.servers = None
        self.name = 'EAS'

    
        
    def get_best_server(self, task) :
        loads = []
        for s in self.servers.values():
            if s.first_check(task):
                loads.append((s, s.expected_power_variation(task)))
        min_load = min(loads, key=lambda x: x[1])[1]

        # To prevent floating point errors
        candidates = [ s for s, l in loads
                    if abs(l - min_load) < 1e-9 ]

        return random.choice(candidates)

    def assign_tasks(self, ready_tasks, t):

        for task in ready_tasks:
            server = self.get_best_server(task)
            server.add_task_to_queue(
                task,
                t
            )