import numpy as np
from collections import deque
try :
    from classes.VmClass import Vm
except ModuleNotFoundError: 
    from VmClass import Vm
    


class Server:
    _server_count = 0
    def __init__(
        self,
        c_cpu: float,
        c_ram: float,
        alpha: float,
        beta: float = None,
        server_farm_id: int = None,
        vms: list = None,
        id: int = None,
        storage: int = 3072,
        static_power: float = 130,
        optimal_utilization_rate: float = 0.75,
        virtualization_level : float = 0.9
    ):
        if id is None :
            self.id = Server._server_count
            Server._server_count +=1 
        else :
            self.id = id
        self.server_farm_id = server_farm_id
        self.vms = self.populate_vm(vms) #create a dict of vms with their ids as keys for faster access
        self.c_cpu = c_cpu
        self.c_ram = c_ram

        self.alpha = alpha
        self.beta = beta

        self.static_power = static_power
        self.optimal_utilization_rate = optimal_utilization_rate
        
        
        self.hosted_tasks = {}
        self.storage = storage
        
        self.task_queue = deque()
        self.virtualization_level = virtualization_level
        
        self.virtual_capacity = self.virtualization_level*self.c_cpu

    def populate_vm(self, vms):
        if vms is None :
            return {}
        for vm in vms:
            vm.server = self
        return {vm.id: vm for vm in vms}

    def spawn_vm(self, vm):
        vm.server = self
        self.vms[vm.id] = vm

    def spawn_vm_group(self, cpu=[32], ram=[32]):

        start_id = max(self.vms.keys()) + 1 if self.vms else 0

        assert len(cpu) == len(ram), "\n[INVALID INPUT]\nCPU, RAM lists must match"
        assert sum(cpu) <= self.c_cpu and sum(ram) <= self.c_ram , f"\n[INVALID INPUT]\nCPU, RAM requirements exceed the server capacity ! {sum(cpu)} <= {self.c_cpu}, {sum(ram)} <= {self.c_ram }"

        for i in range(len(cpu)):
            vm_id = start_id + i
            new_vm = Vm(id=vm_id, c_cpu=cpu[i], c_ram=ram[i])
            self.spawn_vm(new_vm)
        
    
    def is_available(self) -> bool:
        return (self.cpu_utilization() < 1 and
                self.ram_utilization() < 1 and
                self.storage_utilization() < 1
                )
        
        

    # Basic Hosting : first in list, first served => to be enhanced
    def execute_tasks(self, t):
        while self.task_queue:
            task = self.task_queue[0]  # look at first task
            if task.status == 4: # must be pending
                hosted = False
                for vm in self.vms.values():
                    if vm.check_req_constraint(task):
                        if vm.host_task(task, t):
                            self.hosted_tasks[task] = vm
                            task.server = self
                            task.vm = vm
                            self.task_queue.popleft()
                            hosted = True
                            break

                if not hosted:
                    break
            else:
                self.task_queue.popleft()
        
    def add_task_to_queue(self,task, t) :
        if task.size + self.get_storage_usage() <= self.storage :
            task.status = 4 # waiting in the queue
            self.task_queue.append(task)
            task.start_time = t
            return True
        return False

    def first_check(self, task) :
        return (task.size + self.get_storage_usage() <= self.storage)
    
    def release_task_from_server(self, task):
        return self.hosted_tasks[task].release_task()
    
    
    def cpu_utilization(self):
        return np.sum(vm.used_cpu for vm in self.vms.values())/self.c_cpu
    
    def virtual_cpu_efficiency(self):
        return np.sum(vm.used_cpu for vm in self.vms.values())/self.virtual_capacity

    def ram_utilization(self):
        return np.sum(vm.used_ram for vm in self.vms.values())/self.c_ram

    
    
    
    def get_power_consumption(self):
        cpu_utilization = self.cpu_utilization()
        return  round((cpu_utilization**self.beta)*self.alpha + self.static_power, ndigits=3)
    
    
    def get_storage_usage(self):
        return (np.sum(_tsk.size for _tsk in self.hosted_tasks.keys()) +
                np.sum(_tsk.size for _tsk in self.task_queue))
        
    
    def storage_utilization(self):
        return self.get_storage_usage()/self.storage
    
    def time_step_vm(self, _t, time_step=1):
        for vm in self.vms.values():
            if vm.hosted_task is not {}:
                vm.time_step_tasks(_t)
