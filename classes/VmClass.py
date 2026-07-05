class Vm:
    def __init__(self, id: int, c_cpu: float, c_ram: float):

        self.id = id

        self.cpu = c_cpu
        self.ram = c_ram

        self.server_id = None

        # 0 = IDLE, 1 = BUSY
        self.status = 0

        self.hosted_task = None

        self.used_cpu = 0.0
        self.used_ram = 0.0
        
        

    # ----------------------------
    # RESOURCE CHECK
    # ----------------------------
    def check_req_constraint(self, task):
        return (
            self.used_cpu + task.cpu <= self.cpu and
            self.used_ram + task.ram <= self.ram
        )

    # ----------------------------
    # HOST TASK
    # ----------------------------
    def host_task(self, task):
        self.hosted_task = task
        self.status = 1  # busy

        self.used_cpu += task.cpu
        self.used_ram += task.ram

        task.vm_id = self.id
        task.status = 2  # running
        return True

    # ----------------------------
    # FINISH TASK
    # ----------------------------
    def release_task(self):
        if self.hosted_task is None:
            return False

        self.used_cpu -= self.hosted_task.cpu
        self.used_ram -= self.hosted_task.ram

        self.hosted_task.status = 0  # finished
        self.hosted_task.on_finished()

        self.hosted_task = None
        self.status = 0  # idle again

        return True