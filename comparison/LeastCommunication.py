from templates.Scheduler import Scheduler


class DataLocalityAwareScheduler(Scheduler):

    def __init__(self, mode = 'NAIVE'):
        super().__init__('DLAS' + '-' +mode)
        self.mode = mode
        assert mode in ['NAIVE', 'HYBRID'],\
            "[BAD INPUT] AVAILABLE MODES : NAIVE | HYBRID"
        
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


    def assign_tasks(self, ready_tasks, t):
        
        match self.mode:
            case 'NAIVE' :
                pairs = self.get_best_pairs(ready_tasks)
            case 'HYBRID' :
                pairs = self.get_best_pairs_hybrid(ready_tasks)

        for task, server in pairs.items():

            server.add_task_to_queue(
                task,
                t
            )