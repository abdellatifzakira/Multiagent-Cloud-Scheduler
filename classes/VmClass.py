from collections import deque

class Vm:
    def __init__(self, id: int, c_cpu: float, c_ram: float, compute_power: float):

        self.id = id

        self.cpu = c_cpu
        self.ram = c_ram
        self.compute_power = compute_power


        self.server = None

        self.hosted_task = {}
        self.pending_tasks = deque()

        self.used_cpu = 0.0
        self.used_ram = 0.0
        
        self.max_concurrent_tasks = 25
        self.completed_tasks = 0
        

    # ----------------------------
    # RESOURCE CHECK
    # ----------------------------
    def check_req_constraint(self, task):
        return (
            self.used_cpu + task.cpu <= self.cpu and
            self.used_ram + task.ram <= self.ram and
            len(list(self.hosted_task.values())) + 1 <= self.max_concurrent_tasks
        )
        

        


    # ----------------------------
    # HOST TASK
    # ----------------------------
    def host_task(self, task):     
        
        self.hosted_task[task] = task.id

        self.used_cpu += task.cpu
        self.used_ram += task.ram
        
        task.status = 2  # running
        
        return True
    
    
    def time_step_tasks(self,_t, time_step) :
        
        if len(self.hosted_task.keys()) > 0 :
            actual_instruction_rate = time_step*self.compute_power
            actual_instruction_rate_per_task = actual_instruction_rate/len(self.hosted_task.keys())
        
        if self.hosted_task is {} :
            return
        for _task in list(self.hosted_task.keys()):
            _task.remaining_instructions -= actual_instruction_rate_per_task
            _task.advance_timer(_t, time_step)

    # ----------------------------
    # FINISH TASK
    # ----------------------------
    def release_task(self,task, t):
        if task not in list(self.hosted_task.keys()):
            return False

        self.used_cpu -= task.cpu
        self.used_ram -= task.ram
        
        task.on_finished(t) # Inter-tasks notification 
        task.end_time = t
        task.status = 0

        
        """
        print("task finished", task.id, task.job_id)
        for child in task.children:
            print("child ready", child.id, child.job_id)
        """
        if self.server.network_enabled:
            self.server.save_data(task, t)

        self.hosted_task.pop(task)
        self.server.hosted_tasks.pop(task)
        self.completed_tasks += 1

        return True
    