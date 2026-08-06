import numpy as np
from collections import deque
try :
    from classes.VmClass import Vm
except ModuleNotFoundError: 
    from VmClass import Vm
from utilities.helpers import parse_power_equation


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
        mode : str = 'ROUNDROBIN',
    ):
        assert mode in ['ROUNDROBIN', 'LEAST_LOADED', 'EXECUTION_TIME', 'COLLABORATIVE'], \
            "[BAD INPUT] : AVAILABLE MODES ROUNDROBIN | LEAST_LOADED | EXECUTION_TIME | COLLABORATIVE"
        
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
        self.index = 0
        self.num_vms = 0
        self.vm_id_list = list(self.vms.keys())
        
        self.power_model = 'DEFAULT'
        self.power_function = self.dynamic_power
        
        
    
    def set_power_model(self, model):
        self.power_model = model
        if model != 'DEFAULT' :
            self.power_function = parse_power_equation(self.power_model)
    
    def receive_data(self, duo, data):
        try :
            self.received_data[duo] += data
        except KeyError :
            self.received_data[duo] = data
            
            
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
        epsilon = 1e-9
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
            # adding epsilon to count for floating point errors
            not_received = (total_received_data + saved_data + epsilon < required_data[parent])
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
        self.num_vms += 1
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
        self.vm_id_list = list(self.vms.keys()) 
    
    def roundrobin(self, task):
        if not self.vm_id_list:
            return False
        # for single queued tasks run first fit to prevent infinite loops
        if len(self.task_queue) == 1:
            for vm in self.vms.values():
                if vm.check_req_constraint(task):
                    if vm.host_task(task):
                        self.hosted_tasks[task] = vm
                        task.server = self
                        task.vm = vm
                        return True
                
        # for multiple tasks run roundrobin
        target_vm_id = self.vm_id_list[self.index]
        vm = self.vms[target_vm_id]
        
        if vm.check_req_constraint(task):
            if vm.host_task(task):
                self.hosted_tasks[task] = vm
                task.server = self
                task.vm = vm
                self.index = (self.index + 1) % len(self.vm_id_list)
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
        # for single queued tasks run first fit to prevent infinite loops
        if len(self.task_queue) == 1:
            for vm in self.vms.values():
                if vm.check_req_constraint(task):
                    if vm.host_task(task):
                        self.hosted_tasks[task] = vm
                        task.server = self
                        task.vm = vm
                        return True
        def get_expected_latency(vm, new_task = task):
            vm = vm[1]

            tasks = list(vm.hosted_task.keys()) + [new_task]

            total_cpu = sum(
                t.cpu for t in tasks
            )

            finish_times = []

            for t in tasks:

                share = t.cpu / total_cpu

                effective_power = (
                    vm.compute_power * share
                )
                
                if effective_power <= 0 :
                    raise ValueError("0 compute power detected ")

                finish_times.append(
                    t.remaining_instructions / effective_power
                )

            return sum(finish_times)/len(finish_times)
        
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
        if not self.task_queue:
            return
        unhosted_buffer = deque()
        while self.task_queue:
            task = self.task_queue.popleft()
            if task.status == 4 and not task.scheduled:
                if self.check_data_availability(task) : 
                    if task.time_data_arrival is None:
                        task.time_data_arrival = t
                        
                    hosted = False
                    if self.mode == 'ROUNDROBIN':
                        hosted = self.roundrobin(task)
                    elif self.mode == 'LEAST_LOADED':
                        hosted = self.least_loaded(task)
                    elif self.mode == 'EXECUTION_TIME':
                        hosted = self.fastest_vm(task)
                    
                    if hosted:
                        task.scheduled = True
                    else:
                        unhosted_buffer.append(task)
                else:
                    unhosted_buffer.append(task)
            else:
                pass

        # Push delayed allocations back to the main active queue
        self.task_queue = unhosted_buffer
        
    def add_task_to_queue(self,task, t) :
        task.status = 4 # waiting in the queue
        self.task_queue.append(task)
        task.server = self
        task.start_time = t
        return True

    def first_check(self, task) :
        verdict = ((task.size + self.get_storage_usage() <= self.storage) and
                   (task.cpu <= max([vm.cpu for vm in self.vms.values()])) and
                   (task.ram <= max([vm.ram for vm in self.vms.values()])))
        return verdict

    
    def cpu_utilization(self):
        return np.sum(vm.used_cpu for vm in self.vms.values())/self.c_cpu
    
    def virtual_cpu_efficiency(self):
        usage = np.sum(vm.used_cpu for vm in self.vms.values())/self.virtual_capacity
        self.peak_cpu = max(self.peak_cpu, usage)
        return usage

    def ram_utilization(self):
        return np.sum(vm.used_ram for vm in self.vms.values())/self.c_ram

    
    
    
    def get_power_consumption(self):
        if self.power_model == 'DEFAULT' :
            cpu_utilization = self.cpu_utilization()
            return  round((cpu_utilization**self.beta)*self.alpha + self.static_power, ndigits=3)
        else :
            CPU = self.cpu_utilization()
            RAM = self.ram_utilization()
            STORAGE = self.storage_utilization()
            return round(self.power_function(CPU, RAM, STORAGE) + self.static_power, ndigits=3)
    
    def dynamic_power(self,cpu, ram, storage):
        return  round((cpu**self.beta)*self.alpha , ndigits=3)
            
    
    
    def get_cpu_usage_variation(self, dcpu):
        "get the cpu percentage variation caused by a task required cpu"
        return dcpu/self.c_cpu
    
    def get_ram_usage_variation(self, dram):
            "get the ram percentage variation caused by a task required ram"
            return dram/self.c_ram
    
    def get_storage_usage_variation(self, dstorage):
            "get the storage percentage variation caused by a task required storage"
            return dstorage/self.storage
    
    def expected_power_variation(self, task) :
        dcpu = self.get_cpu_usage_variation(task.cpu)
        dram = self.get_ram_usage_variation(task.ram)
        dstorage = self.get_storage_usage_variation(task.size)
        CPU = self.cpu_utilization() + dcpu
        RAM = self.ram_utilization() + dram
        STORAGE = self.storage_utilization() + dstorage
        return round(self.power_function(CPU, RAM, STORAGE) + self.static_power, ndigits=3)

    
    
    def get_storage_usage(self):
        return (np.sum(_tsk.size for _tsk in self.hosted_tasks.keys()) +
                np.sum(_tsk.size for _tsk in self.task_queue))
        
    
    def storage_utilization(self):
        return self.get_storage_usage()/self.storage
    
    def time_step_vm(self, _t, time_step):
        for vm in self.vms.values():
            if vm.hosted_task is not {}:
                vm.time_step_tasks(_t, time_step)
