import random
from templates.Scheduler import Scheduler

class EnergyAwareScheduler(Scheduler):

    def __init__(self):
        super().__init__('EAS', 'Energy aware scheduler')
    
        
    def get_best_server(self, task) :
        loads = []
        for s in self.servers.values():
            if s.first_check(task):
                loads.append((s, s.expected_power_variation(task)))
        if len(loads) <= 0:
            return
        min_load = min(loads, key=lambda x: x[1])[1]

        # To prevent floating point errors
        candidates = [ s for s, l in loads
                    if abs(l - min_load) < 1e-9 ]

        return self.rng.choice(candidates)

    def assign_tasks(self, ready_tasks, t):

        for task in ready_tasks:
            server = self.get_best_server(task)
            if server is None :
                continue
            server.add_task_to_queue(
                task,
                t
            )