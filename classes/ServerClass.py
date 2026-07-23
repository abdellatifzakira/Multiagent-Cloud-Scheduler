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
        
        self.tasks_waiting_data = []
        self.outgoing_data = {}
        self.received_data = {}
        self.saved_data = {}
        
        self.network_enabled = None
    
    def request_data(self):
        requested_data = []
        for child in self.waiting_queue :
            if not child.data_requested :
                requested_data.append(child)
                child.data_requested = True
        return requested_data
            
            
    def save_data(self, task, t):
        for child in task.children :
            # (sender, receiver) : [sent data, time, status: 0 -> pending, 1 -> sent]
            self.saved_data[(task, child)] = [child.parent_weights[task], t, 0]
        
    
    def send_data(self):
        ready_payloads = {key: value for key, value in self.saved_data.items() if value[2] == 0 and
                          key[1].server is not None and
                          key[1].server != key[0].server }
        
        for k, v in ready_payloads.items() :
            self.outgoing_data[k] = v
            self.saved_data.pop(k)
        
        if ready_payloads:
            for value in ready_payloads.values():
                value[2] = 1 # changing the status
        return ready_payloads
    
    
    def has_data(self):
        ready_payloads = {key: value for key, value in self.saved_data.items() if value[2] == 0 and
                          key[1].server is not None and
                          key[1].server != key[0].server}
        return ready_payloads != {}

        
    
    def check_data_availability(self, task):
        if not self.network_enabled:
            return True
        required_data = task.parent_weights
        for parent in required_data.keys():
            total_received_data = 0
            saved_data = 0
            if (parent, task) in self.received_data.keys() :
                total_received_data = self.received_data[(parent, task)]
            if (parent, task) in self.saved_data.keys() :
                saved_data = self.saved_data[(parent, task)][0]
            not_received = (total_received_data + saved_data < required_data[parent])
            if not_received :
                    return False
        return True
        

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
            if task.status == 4 : # must be pending
                if self.check_data_availability(task = task) and not task.scheduled:
                    if  task.time_data_arrival  is None :
                        task.time_data_arrival = t
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
                        task.scheduled = True
                        _idx = 0 
                    else :
                        _idx +=1
                else :
                    if task not in self.tasks_waiting_data :
                        self.tasks_waiting_data.append(task)
                    _idx +=1
            else:
                self.task_queue.popleft()
        
    def add_task_to_queue(self,task, t) :
        if task.size + self.get_storage_usage() <= self.storage :
            task.status = 4 # waiting in the queue
            self.task_queue.append(task)
            task.server = self
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
    
    
    """
    >> the new energy consumption after a cpu variation of "e" assuming "e" is minimal "e<<cpu"
    >> E(cpu + e) = E(cpu) + e * dE/dcpu (cpu)
    >> E(cpu + e) = E(cpu) + e * alpha * beta * (cpu^(beta -1))
    >> the marginal power consumption : e * alpha * beta * (cpu^(beta -1))
    """
    def get_cpu_usage_variation(self, dcpu):
        "get the cpu percentage variation caused by a task requirement cpu"
        return dcpu/self.c_cpu
    
    def expected_power_variation(self, task) :
        dcpu = self.get_cpu_usage_variation(task.cpu)
        expected_power = dcpu * self.alpha * self.beta * (self.cpu_utilization()**(self.beta -1))
        
        return self.get_power_consumption() + expected_power
    

    
    
    def get_storage_usage(self):
        return (np.sum(_tsk.size for _tsk in self.hosted_tasks.keys()) +
                np.sum(_tsk.size for _tsk in self.task_queue))
        
    
    def storage_utilization(self):
        return self.get_storage_usage()/self.storage
    
    def time_step_vm(self, _t, time_step):
        for vm in self.vms.values():
            if vm.hosted_task is not {}:
                vm.time_step_tasks(_t, time_step)
