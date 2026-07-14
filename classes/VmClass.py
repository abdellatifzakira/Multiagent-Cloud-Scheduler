from collections import deque
class Vm:
    def __init__(self, id: int, c_cpu: float, c_ram: float, c_storage: float):

        self.id = id

        self.cpu = c_cpu
        self.ram = c_ram
        self.storage = c_storage

        self.server = None

        # 0 = IDLE, 1 = BUSY
        #self.status = 0

        self.hosted_task = {}
        self.pending_tasks = deque()

        self.used_cpu = 0.0
        self.used_ram = 0.0
        self.used_storage = 0.0
        
        self.max_concurrent_tasks = 5
        
        #self.timer = 0.0
        

    # ----------------------------
    # RESOURCE CHECK
    # ----------------------------
    def check_req_constraint(self, task):
        return (
            self.used_cpu + task.cpu <= self.cpu and
            self.used_ram + task.ram <= self.ram and
            self.used_storage + task.size <= self.storage and
            len(list(self.hosted_task.values())) + 1 <= self.max_concurrent_tasks
        )


    # ----------------------------
    # HOST TASK
    # ----------------------------
    def host_task(self, task, t):
        self.hosted_task[task] = task.id
        
        # Initialize the task internal timer 
        task.timer = task.runtime
        
        #self.status = 1  # busy

        self.used_cpu += task.cpu
        self.used_ram += task.ram
        self.used_storage += task.size
        
        task.status = 2  # running
        task.start_time = t
        #self.timer = task.runtime
        
        return True
    
    
    def time_step_tasks(self,_t) :
        if self.hosted_task is {} :
            return
        for _task in list(self.hosted_task.keys()):
            _task.advance_timer(_t)

    # ----------------------------
    # FINISH TASK
    # ----------------------------
    def release_task(self,task, t):
        if task not in list(self.hosted_task.keys()):
            return False

        self.used_cpu -= task.cpu
        self.used_ram -= task.ram
        self.used_storage -= task.size
        
        task.on_finished(t) # Inter-tasks notification 
        task.status = 0
        task.end_time = t

        self.hosted_task.pop(task)
        self.server.hosted_tasks.pop(task)
    
        
        #self.status = 0  # idle again
        
        #self.timer = 0.0

        return True