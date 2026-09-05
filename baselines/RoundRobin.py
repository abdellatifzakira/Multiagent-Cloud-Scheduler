import numpy as np
from templates.Scheduler import Scheduler

class RoundRobinScheduler(Scheduler):
    def __init__(self, mode="BENCHMARK"):
        super().__init__("RR", 'Round Robin')
        self.pointer = 0
        self.mode = mode
        
    
        
    def assign_tasks(self, ready_tasks, t):
        n = len(self.servers)
        for task in ready_tasks :
            assert all(parent.status == 0 for parent in task.parents),(
                    f"DAG violation: Task {task.id} scheduled "
                    "before all parents finished"
                )
            server = self.servers[self.pointer]
            success = False
            if server.first_check(task):
                success = server.add_task_to_queue(task, t)
            if success or self.mode == "CHECK":
                self.pointer = (self.pointer + 1) % n


