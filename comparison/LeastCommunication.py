from templates.Scheduler import Scheduler
import numpy as np


class DataLocalityAwareScheduler(Scheduler):

    def __init__(self, mode = 'NAIVE'):
        super().__init__('DLAS' + '-' + mode, 'Data Locality Aware Scheduler : mode ' + mode)
        self.mode = mode
        assert mode in ['NAIVE', 'HYBRID', 'BALANCED'],\
            "[BAD INPUT] AVAILABLE MODES : NAIVE | HYBRID | BALANCED"
        
        self.index = 0



    def get_best_pairs(self, tasks):
        best_pairs = {}
        for task in tasks:
            if len(task.parents) <= 0 :
                for server in self.servers.values():
                    if server.first_check(task) :
                        best_pairs[task] = server
                        break
            else:
                assert all(parent.status == 0 for parent in task.parents),(
                                f"DAG violation: Task {task.id} scheduled "
                                "before all parents finished"
                            )
                
                parents_weights = sorted(
                                    task.parent_weights.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                        )
                for parent, weight in parents_weights :
                    if parent.server.first_check(task) :
                        best_pairs[task] = parent.server
                        break
        return best_pairs

    def get_best_pairs_hybrid(self, tasks):
        best_pairs = {}
        for task in tasks:
            # for parent tasks run RR
            if len(task.parents) <= 0 :
                server = self.servers[self.index]
                if server.first_check(task) :
                    best_pairs[task] = server
                    self.index = (self.index + 1) % len(list(self.servers.values()))
                    continue
            else:
                assert all(parent.status == 0 for parent in task.parents),(
                                f"DAG violation: Task {task.id} scheduled "
                                "before all parents finished"
                            )
                
                parents_weights = sorted(
                                    task.parent_weights.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                        )
                for parent, weight in parents_weights :
                    if parent.server.first_check(task) :
                        best_pairs[task] = parent.server
                        break
        return best_pairs
    
    
    def get_best_pairs_balanced(self, tasks):
        best_pairs = {}
        for task in tasks:
            # for parent tasks run LL
            if len(task.parents) <= 0 :
                servers = self.servers.values()
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
                min_load = min(loads, key=lambda x: x[1])[1]
                
                # To prevent floating point errors
                candidates = [ s for s, l in loads
                                    if abs(l - min_load) < 1e-9 ]
                
                best_pairs[task] = self.rng.choice(candidates)
            else:
                assert all(parent.status == 0 for parent in task.parents),(
                                f"DAG violation: Task {task.id} scheduled "
                                "before all parents finished"
                            )
                
                parents_weights = sorted(
                                    task.parent_weights.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                        )
                for parent, weight in parents_weights :
                    if parent.server.first_check(task) :
                        best_pairs[task] = parent.server
                        break
        return best_pairs
    



    def assign_tasks(self, ready_tasks, t):
        
        match self.mode:
            case 'NAIVE' :
                pairs = self.get_best_pairs(ready_tasks)
            case 'HYBRID' :
                pairs = self.get_best_pairs_hybrid(ready_tasks)
            case 'BALANCED' :
                pairs = self.get_best_pairs_balanced(ready_tasks)

        for task, server in pairs.items():

            server.add_task_to_queue(
                task,
                t
            )