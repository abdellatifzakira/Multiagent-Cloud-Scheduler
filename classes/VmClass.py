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

        self.used_cpu = 0.0
        self.used_ram = 0.0
        self.used_storage = 0.0
        
        #self.timer = 0.0
        

    # ----------------------------
    # RESOURCE CHECK
    # ----------------------------
    def check_req_constraint(self, task):
        return (
            self.used_cpu + task.cpu <= self.cpu and
            self.used_ram + task.ram <= self.ram and
            self.used_storage + task.size <= self.storage
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
        if self.hosted_task is {} or self.hosted_task[task] is None :
            return False

        self.used_cpu -= task.cpu
        self.used_ram -= task.ram
        self.used_storage -= task.size

        task.status = 0  # Finished
        task.end_time = t
        task.on_finished(t) # Inter-tasks notification 

        self.hosted_task.pop(task)
        
        #self.status = 0  # idle again
        
        #self.timer = 0.0

        return True