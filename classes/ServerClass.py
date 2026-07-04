import math
from VmClass import Vm
from TaskClass import Task

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

    def populate_vm(self, vms):
        if vms is None :
            return {}
        for vm in vms:
            vm.server_id = self.id
        return {vm.id: vm for vm in vms}

    def spawn_vm(self, vm):
        vm.server_id = self.id
        self.vms[vm.id] = vm

    def spawn_vm_group(self, cpu=[0.5], ram=[0.5]):

        start_id = max(self.vms.keys()) + 1 if self.vms else 0

        assert len(cpu) == len(ram), "\n[INVALID INPUT]\nCPU and RAM lists must match"
        assert sum(cpu) <= self.c_cpu and sum(ram) <= self.c_ram , "\n[INVALID INPUT]\nCPU or RAM requirements exceed the server capacity !"

        for i in range(len(cpu)):
            vm_id = start_id + i
            new_vm = Vm(id=vm_id, c_cpu=cpu[i], c_ram=ram[i])
            self.spawn_vm(new_vm)
        
    
    def get_idle_vms(self):
        return [vm for vm in self.vms.values() if vm.status == 0]
    
    def get_busy_vms(self):
        return [vm for vm in self.vms.values() if vm.status == 1]
    
    def is_available(self):
        return len(self.get_idle_vms()) > 0

    def host_task_in_server(self, task):
        for vm in self.get_idle_vms():
            if vm.check_req_constraint(task):
                self.hosted_tasks[(task.id, task.job_id)] = vm.id
                return vm.host_task(task)
        return False
    
    def release_task_from_server(self, task_key):
        hosting_vm = self.vms[self.hosted_tasks[task_key]]
        return hosting_vm.release_task()
    
    def cpu_utilization(self):
        vms =  self.vms.values()
        return sum(vm.used_cpu for vm in vms)
    
    
    # at first we try simple linear power consumption
    def get_power_consumption(self):
        cpu_utilization = self.cpu_utilization()
        return round(cpu_utilization*self.alpha + self.static_power, ndigits=3)


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
    
    task_1 =  server.host_task_in_server(task=task_1)
    for vm in server.get_idle_vms() :
        print(f"VM Number : {vm.id} is idle") # expected something like 1,2
    print(server.is_available()) # expected True 
    print(f"{task_1.status = }") # expecting 2
    
    
    task_2 = Task(
        id = 1,
        job_id= 0,
        cpu=0.1,
        ram=0.1,
        status=3,
        runtime=20
    )
    
    task_2 =  server.host_task_in_server(task=task_2)
    for vm in server.get_idle_vms() :
        print(f"VM Number : {vm.id} is idle") # expected something like 2
        
    print(f"Total CPU Utiliazation = {server.cpu_utilization()} | Available CPU capcity = {server.c_cpu}")
    
    print(f"Total Power Consumption = {server.get_power_consumption()}")
    
    print("RELEASING A TASK, FINISHING IT")
    task_finished = server.release_task_from_server((0,0))
    
    print(f"{task_finished.status = }")
    
    for vm in server.get_idle_vms() :
        print(f"VM Number : {vm.id} is idle") # expected something 0, 2
    
    
    