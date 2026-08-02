import numpy as np
import random
from templates.Scheduler import Scheduler
class LeastLoadedScheduler(Scheduler):
    def __init__(self,
                 mode : str = 'CPU',
                 sorting : str = 'FIFO'
                 ):
        super().__init__('LL' + '-' + mode, 'Least Loaded Scheduler : mode ' + mode + ' | sorting ' + sorting)
        self.mode = mode
        self.sorting = sorting

        
        assert mode in ['CPU', 'RAM', 'QUEUE', 'HYBRID'], "[INVALID MODE]\nAVAILABLE MODES : CPU | RAM | QUEUE | HYBRID "
        assert sorting in ['FIFO', 'CPU', 'SLA'], "[INVALID INPUT]\nAVAILABLE SORTING PARAMETERS : FIFO | CPU | SLA"


    def sort_tasks(self, ready_tasks, current_time):
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
            case 'SLA' :
                ready_tasks.sort(
                    key=lambda task: task.job_sla - (current_time - task.job_arrival)
                )
                
                
    def expected_server_latency(self,server):

        # queued work
        queued_instructions = sum(
            task.num_instructions 
            for task in server.task_queue
        )

        total_compute = sum(
            vm.compute_power 
            for vm in server.vms.values()
        )

        queue_delay = (
            queued_instructions / total_compute
            if total_compute > 0
            else float("inf")
        )


        # current running tasks
        vm_finish_times = []

        for vm in server.vms.values():

            remaining = sum(
                task.remaining_instructions
                for task in vm.hosted_task.keys()
            )

            vm_time = remaining / vm.compute_power

            vm_finish_times.append(vm_time)

        # when the first slot becomes available
        execution_delay = max(vm_finish_times)

        # weights to be tuned
        return 0.75*queue_delay + 0.25*execution_delay
    
    def get_least_busy_server(self):
        servers = list(self.server_farm.servers.values())

        match self.mode:
            case 'QUEUE' :
                """
                the waiting time in the queue + the remaining execution time
                of the hosted asks all divided by the parallelism capacity of the server
                """
                loads = [
                            (s, self.expected_server_latency(s))
                            for s in servers
                        ]
            case 'CPU' :
                """
                a score based on the actual cpu usage,
                max vm cpu availability, and queue cpu req pressure
                """
                loads = []
                for s in servers:
                    current = s.virtual_cpu_efficiency()

                    max_available_vm_cpu = sum(
                        vm.cpu - vm.used_cpu
                        for vm in s.vms.values()
                    )
                    
                    queue_pressure = np.sum(
                        [tsk.cpu for i, tsk in enumerate(s.task_queue)]
                    )
                    queue_pressure_norm = max(0, queue_pressure - max_available_vm_cpu)/(s.c_cpu*s.virtualization_level)
                    
                    score = current + queue_pressure_norm

                    loads.append((s, score))

            case 'RAM' :
                loads = [(s, s.ram_utilization()) for s in servers]
                
            
            case 'HYBRID' :
                loads = []
                for s in servers:
                    current = s.cpu_utilization()

                    max_available_vm_cpu = max(
                        vm.cpu - vm.used_cpu
                        for vm in s.vms.values()
                    )
                    
                    queue_pressure = np.sum(
                        [tsk.cpu for i, tsk in enumerate(s.task_queue)]
                    )
                    score_cpu = current + max(
                        0,
                        queue_pressure - max_available_vm_cpu
                    )
                    
                    score_queue = (sum(tsk.runtime for tsk in s.task_queue) + sum(tsk.timer for tsk in s.hosted_tasks.keys()))/sum(vm.max_concurrent_tasks for vm in s.vms.values())

                    loads.append((s, 0.3*score_cpu + 0.7*score_queue))
                

        min_load = min(loads, key=lambda x: x[1])[1]

        # To prevent floating point errors
        candidates = [ s for s, l in loads
                    if abs(l - min_load) < 1e-9 ]

        return self.rng.choice(candidates)
        
        
        
    def assign_tasks(self, ready_tasks, t):
        self.sort_tasks(ready_tasks=ready_tasks,  current_time=t)
        for task in ready_tasks :
            assert all(parent.status == 0 for parent in task.parents),(
                            f"DAG violation: Task {task.id} scheduled "
                            "before all parents finished"
                        )
            server = self.get_least_busy_server()
            server.add_task_to_queue(task, t)


