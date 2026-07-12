import numpy as np

try :
    from classes.VmClass import Vm
    from classes.TaskClass import Task
    from classes.JobClass import Job
except ModuleNotFoundError: 
    from VmClass import Vm
    from TaskClass import Task
    from JobClass import Job
    


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
        static_power: float = 0.035,
        optimal_utilization_rate: float = 0.75
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

    def populate_vm(self, vms):
        if vms is None :
            return {}
        for vm in vms:
            vm.server = self
        return {vm.id: vm for vm in vms}

    def spawn_vm(self, vm):
        vm.server = self
        self.vms[vm.id] = vm

    def spawn_vm_group(self, cpu=[0.5], ram=[0.5], storage = [512]):

        start_id = max(self.vms.keys()) + 1 if self.vms else 0

        assert len(storage) == len(cpu) == len(ram), "\n[INVALID INPUT]\nCPU, RAM and STORAGE lists must match"
        assert sum(storage)<= self.storage and sum(cpu) <= self.c_cpu and sum(ram) <= self.c_ram , "\n[INVALID INPUT]\nCPU, RAM or STORAGE requirements exceed the server capacity !"

        for i in range(len(cpu)):
            vm_id = start_id + i
            new_vm = Vm(id=vm_id, c_cpu=cpu[i], c_ram=ram[i], c_storage= storage[i])
            self.spawn_vm(new_vm)
        
    
    def get_idle_vms(self):
        return [vm for vm in self.vms.values() if vm.status == 0]
    
    def get_busy_vms(self):
        return [vm for vm in self.vms.values() if vm.status == 1]
    
    def is_available(self) -> bool:
        return (self.cpu_utilization() < self.c_cpu and
                self.ram_utilization() < self.c_ram and
                self.storage_utilization() < self.storage
                )
        
        

    # Basic Hosting : first in list, first served => to be enhanced
    def host_task_in_server(self, task, t):
        if task.status != 1 :
            return False
        for vm in self.vms.values():
            if vm.check_req_constraint(task):
                self.hosted_tasks[task] = vm
                task.server = self
                task.vm = vm
                return vm.host_task(task, t)
        return False
    
    def release_task_from_server(self, task):
        return self.hosted_tasks[task].release_task()
    
    
    def cpu_utilization(self):
        return np.sum(vm.used_cpu for vm in self.vms.values())
    
    def ram_utilization(self):
        return np.sum(vm.used_ram for vm in self.vms.values())
    def storage_utilization(self):
        return np.sum(vm.used_storage for vm in self.vms.values())
    
    
    # at first we try simple linear power consumption
    def get_power_consumption(self):
        cpu_utilization = self.cpu_utilization()
        return round(cpu_utilization*self.alpha + self.static_power, ndigits=3)
    
    
    def get_storage_usage(self):
        return sum(vm.used_storage for vm in self.vms.values())
    
    def time_step_vm(self, _t, time_step=1):
        for vm in self.vms.values():
            if vm.hosted_task is not {}:
                vm.time_step_tasks(_t)

# QUICK TESTS :

if __name__ == "__main__":
    server = Server(c_ram=1.0, c_cpu=1.0, alpha=0.05)
    server.spawn_vm_group(cpu = [0.3 , 0.3, 0.1], ram=[0.25, 0.25, 0.25])
    
    print(f"VM Number : {vm.id} is idle" for vm in server.get_idle_vms()) # expected 0,1,2
    print(server.is_available()) # expected True 
    
    task_1 = Task(
        id = 0,
        job_id= 0,
        cpu=0.1,
        ram=0.1,
        status=3,
        runtime=20
    )
    
    if server.host_task_in_server(task=task_1) :
        for vm in server.get_idle_vms() :
            print(f"VM Number : {vm.id} is idle") # expected something like 1,2
        print(server.is_available()) # expected True 
        print(f"{task_1.status = }") # expecting 2
    else :
        print("FAILED TO HOST TASK 1 IN SERVER")
    
    
    task_2 = Task(
        id = 1,
        job_id= 0,
        cpu=0.1,
        ram=0.1,
        status=3,
        runtime=20
    )
    
    if server.host_task_in_server(task=task_2) :
        for vm in server.get_idle_vms() :
            print(f"VM Number : {vm.id} is idle") # expected something like 2
    else :
        print("FAILED TO HOST TASK 2 IN SERVER")
        
    print(f"Total CPU Utiliazation = {server.cpu_utilization()} | Available CPU capcity = {server.c_cpu}")
    
    print(f"Total Power Consumption = {server.get_power_consumption()}")
    
    print("RELEASING A TASK 1, FINISHING IT")
    server.release_task_from_server((0,0)) # task 1 released <==> finished
    
    print(f"{task_2.status = }") # expected to be 2 : runining
    print(f"{task_1.status = }") # expected to be 0 : finished
    
    for vm in server.get_idle_vms() :
        print(f"VM Number : {vm.id} is idle") # expected something 0, 2
        
    
    # testing a non-hostable task 
    
    impossible_task = Task(
        id = 1,
        job_id= 0,
        cpu=0.9, # high cpu usage
        ram=0.1,
        status=3,
        runtime=20
    )
    
    if server.host_task_in_server(task=impossible_task) :
        for vm in server.get_idle_vms() :
            print(f"VM Number : {vm.id} is idle") # expected something like 1,2
        print(server.is_available()) # expected True 
    else :
        print(f"FAILED TO HOST TASK IN SERVER !")
    
    print(f"{impossible_task.status = }") # expecting 3
    
    
    
    # testing cross task notification :
    
    for vm in server.get_idle_vms() :
            print(f"VM Number : {vm.id} is idle")
            
            
    
    linked_job = Job()
    
    linked_job = linked_job.spawn_job(
        num_tasks= 4,
        cpu_req= [0.01, 0.03, 0.05, 0.03],
        ram_req= [0.01, 0.01, 0.09, 0.03],
        runtime= [10, 10, 25, 10],
        data_transfer_weights= {
         (0,1) : 1,
         (0,2) : 2,
         (1,3) : 3
        }
    )
    print("linked_job created successfuly")
    
    state = {
        0 : "FINISHED",
        1 : "READY",
        2 : "RUNING",
        3 : "INITIALIZED"
    }
    for task in linked_job.tasks.values() :
        print(f"TASK ID : {task.id} | STATUS : {state[task.status]}")
    print("="*25)
    # trying to host the parent task
    if server.host_task_in_server(linked_job.tasks[0]) :
        for task in linked_job.tasks.values() :
            print(f"TASK ID : {task.id} | STATUS : {state[task.status]}")
    print("="*25)
    # releasing the hosted task : (id = 0, job_id = linked_job id)
    if server.release_task_from_server((0, linked_job.id)) :
        for task in linked_job.tasks.values() :
            print(f"===== TASK ID : {task.id}=====\nChildren : {[c.id for c in task.children]}\nPARENTS : {[p.id for p in task.parents]}\nSTATUS : {state[task.status]}")
    
    
    
        
    
    
    
    
    
    
    
    