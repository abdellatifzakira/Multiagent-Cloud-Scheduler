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
        compute_power: float, 
        alpha: float,
        beta: float = None,
        server_farm_id: int = None,
        vms: list = None,
        id: int = None,
        storage: int = 3072,
        static_power: float = 130,
        optimal_utilization_rate: float = 0.75,
        virtualization_level : float = 0.9,
        mode : str = 'EXECUTION_TIME',
    ):
        assert mode in ['SIMPLE', 'LEAST_LOADED', 'EXECUTION_TIME', 'COLLABORATIVE'], \
            "[BAD INPUT] : AVAILABLE MODES SIMPLE | LEAST_LOADED | EXECUTION_TIME | COLLABORATIVE"
        
        if id is None :
            self.id = Server._server_count
            Server._server_count +=1 
        else :
            self.id = id
        self.server_farm_id = server_farm_id
        self.vms = self.populate_vm(vms) #create a dict of vms with their ids as keys for faster access
        self.c_cpu = c_cpu
        self.c_ram = c_ram
        self.compute_power = compute_power
        self.alpha = alpha
        self.beta = beta
        self.mode =  mode
        self.static_power = static_power
        self.optimal_utilization_rate = optimal_utilization_rate
        
        
        self.hosted_tasks = {}
        self.storage = storage
        
        self.task_queue = deque()
        self.virtualization_level = virtualization_level
        
        self.virtual_capacity = self.virtualization_level*self.c_cpu
        self.peak_cpu = 0

    def populate_vm(self, vms):
        if vms is None :
            return {}
        for vm in vms:
            vm.server = self
        return {vm.id: vm for vm in vms}

    def spawn_vm(self, vm):
        vm.server = self
        self.vms[vm.id] = vm
        
    def spawn_vm_group(self, cpu=[32], ram=[32], compute = [1e6]):

        start_id = max(self.vms.keys()) + 1 if self.vms else 0

        assert len(cpu) == len(ram) == len(compute), "\n[INVALID INPUT]\nCPU, RAM, COMPUTE POWER lists must match"
        assert sum(cpu) <= self.c_cpu and sum(ram) <= self.c_ram and sum(compute) <= self.compute_power, f"\n[INVALID INPUT]\nCPU, RAM, COMPUTE POWER requirements exceed the server capacity ! {sum(cpu)} <= {self.c_cpu}, {sum(ram)} <= {self.c_ram }"

        for i in range(len(cpu)):
            vm_id = start_id + i
            new_vm = Vm(id=vm_id, c_cpu=cpu[i], c_ram=ram[i], compute_power=compute[i])
            self.spawn_vm(new_vm)
        
    
    def is_available(self) -> bool:
        return (self.cpu_utilization() < 1 and
                self.ram_utilization() < 1 and
                self.storage_utilization() < 1
                )
    
    def first_fit(self, task):
        for vm in self.vms.values():
            if vm.check_req_constraint(task):
                if vm.host_task(task):
                    self.hosted_tasks[task] = vm
                    task.server = self
                    task.vm = vm
                    return True
        return False
        
    def least_loaded(self, task):
        vms = sorted(
                    self.vms.items(),
                    key=lambda x: x[1].used_cpu/x[1].cpu,
                    reverse=False
                    )
        
        for id, vm in vms :
            if vm.check_req_constraint(task):
                if vm.host_task(task):
                    self.hosted_tasks[task] = vm
                    task.server = self
                    task.vm = vm
                    return True
        return False
    
    
    def fastest_vm(self, task):
        def get_expected_latency(vm):
            _vm = vm[1]
            _all_instructions = sum(
                tsk.remaining_instructions
                for tsk in _vm.hosted_task.keys()
            )
            
            return _all_instructions/_vm.compute_power
        vms = sorted(
                    self.vms.items(),
                    key=get_expected_latency,
                    reverse=False
                    )
        
        for id, vm in vms :
            if vm.check_req_constraint(task):
                if vm.host_task(task):
                    self.hosted_tasks[task] = vm
                    task.server = self
                    task.vm = vm
                    return True
        return False
            
        
        
    def execute_tasks(self, t):
        _idx = 0
        while self.task_queue:
            try :
                task = self.task_queue[_idx]  # look at first task
            except IndexError :
                break
            if task.status == 4: # must be pending
                match self.mode :
                    case 'SIMPLE' :
                        hosted  = self.first_fit(task=task)
                    case 'LEAST_LOADED' :
                        hosted  = self.least_loaded(task=task)
                    case 'EXECUTION_TIME' :
                        hosted  = self.fastest_vm(task=task)
                
                if hosted :
                    self.task_queue.rotate(-_idx)
                    self.task_queue.popleft()
                    self.task_queue.rotate(_idx)
                    _idx = 0 
                else :
                    _idx +=1
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
        usage = np.sum(vm.used_cpu for vm in self.vms.values())/self.virtual_capacity
        self.peak_cpu = max(self.peak_cpu, usage)
        return usage

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
    
    def time_step_vm(self, _t, time_step):
        for vm in self.vms.values():
            if vm.hosted_task is not {}:
                vm.time_step_tasks(_t, time_step)
