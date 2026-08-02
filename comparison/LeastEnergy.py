import random
from templates.Scheduler import Scheduler
import  numpy as np

class EnergyAwareScheduler(Scheduler):

    def __init__(self):
        super().__init__('EAS', 'Energy aware scheduler')
    
        
    def get_best_server(self, task) :
        loads = []
        for s in self.servers.values():
            if not s.first_check(task):
                continue

            avg_compute = np.mean([vm.compute_power for vm in s.vms.values()])

            # --- Queue pressure (sequential) ---
            queue_pressure = sum(
                s.expected_power_variation(tsk) *
                (tsk.num_instructions / avg_compute)
                for tsk in s.task_queue
            )

            # --- Running tasks (parallel) ---
            parallel_factor = len(list(s.hosted_tasks.keys())) or 1
            effective_compute = avg_compute / parallel_factor
            mean_execution_time = 0
            if len(list(s.hosted_tasks.keys())) :
                mean_execution_time = np.mean([(tsk.remaining_instructions / effective_compute) for tsk in s.hosted_tasks.keys()])
            execution_pressure = s.get_power_consumption() * mean_execution_time
            

            # --- New task ---
            queue_delay = sum(
                tsk.num_instructions / avg_compute
                for tsk in s.task_queue
            )

            parallel_factor_new = len(s.hosted_tasks) + 1
            effective_compute_new = avg_compute / parallel_factor_new

            exec_time = task.num_instructions / effective_compute_new

            task_pressure = s.expected_power_variation(task) * (queue_delay + exec_time)

            loads.append((s, execution_pressure + queue_pressure + task_pressure))
        if len(loads) <= 0:
            return
        min_load = min(loads, key=lambda x: x[1])[1]

        # To prevent floating point errors
        candidates = [ s for s, l in loads
                    if abs(l - min_load) < 1e-9 ]
        if len(candidates) <= 0:
            print(min_load)
            raise ValueError("EMPTY CANDIDATES") 

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