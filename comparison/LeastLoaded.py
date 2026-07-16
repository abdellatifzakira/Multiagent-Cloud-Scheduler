import numpy as np
import random
import math
class LeastLoadedScheduler:
    def __init__(self,
                 mode : str = 'CPU',
                 sorting : str = 'FIFO'
                 ):
        self.server_farm = None
        self.servers = None
        self.mode = mode
        self.sorting = sorting

        
        assert mode in ['CPU', 'RAM', 'QUEUE'], "[INVALID MODE]\nAVAILABLE MODES : CPU | RAM | QUEUE "
        assert sorting in ['FIFO', 'CPU', 'RUNTIME'], "[INVALID INPUT]\nAVAILABLE SORTING PARAMETERS : FIFO | CPU | RUNTIME"


    def sort_tasks(self, ready_tasks):
         # sort by arrival time
        match self.sorting :
            case 'FIFO' :
                ready_tasks.sort(
                    key=lambda task: task.arrival_time
                )
            case 'CPU' :
                ready_tasks.sort(
                    key=lambda task: task.cpu
                )
            case 'RUNTIME' :
                ready_tasks.sort(
                    key=lambda task: task.runtime
                )
    
    def get_least_busy_server(self):
        servers = list(self.server_farm.servers.values())

        match self.mode:
            case 'QUEUE' :
                """
                the waiting time in the queue + the remaining execution time
                of the hosted asks all divided by the parallelism capacity of the server
                """
                loads = [
                    (s, (sum(tsk.runtime for tsk in s.task_queue) + sum(tsk.timer for tsk in s.hosted_tasks.keys()))/sum(vm.max_concurrent_tasks for vm in s.vms.values())
                     )
                    for s in servers
                ]
            case 'CPU' :
                """
                a score based on the actual cpu usage,
                max vm cpu availability, and queue cpu req pressure
                """
                loads = []
                for s in servers:
                    current = s.cpu_utilization()

                    max_available_vm_cpu = max(
                        vm.cpu - vm.used_cpu
                        for vm in s.vms.values()
                    )
                    
                    # A weighted queue pressure, assuming front tasks are more likely to get assigned,
                    # therfore more likely to surge the vm/server
                    queue_pressure = np.sum(
                        [tsk.cpu*math.exp(-0.3*i) for i, tsk in enumerate(s.task_queue)]
                    )
                    score = current + max(
                        0,
                        queue_pressure - max_available_vm_cpu
                    )

                    loads.append((s, score))

            case 'RAM' :
                loads = [(s, s.ram_utilization()) for s in servers]

        min_load = min(loads, key=lambda x: x[1])[1]

        # To prevent floating point errors
        candidates = [ s for s, l in loads
                    if abs(l - min_load) < 1e-9 ]

        return random.choice(candidates)
        
        
        
    def assign_tasks(self, ready_tasks, t):
        self.sort_tasks(ready_tasks=ready_tasks)
        for task in ready_tasks :
            server = self.get_least_busy_server()
            server.add_task_to_queue(task, t)


