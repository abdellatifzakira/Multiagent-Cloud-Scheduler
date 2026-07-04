import math
from classes.VmClass import Vm

class Server:
    def __init__(
        self,
        id: int,
        server_farm_id: int,
        vms: list,
        c_cpu: float,
        c_ram: float,
        alpha: float,
        beta: float,
        static_power: float = 0.035,
        optimal_utilization_rate: float = 0.75
    ):

        self.id = id
        self.server_farm_id = server_farm_id

        self.vms = self.populate_vm(vms) #create a dict of vms with their ids as keys for faster access

        self.c_cpu = c_cpu
        self.c_ram = c_ram

        self.alpha = alpha
        self.beta = beta

        self.static_power_value = static_power
        self.optimal_utilization_rate = optimal_utilization_rate

    def populate_vm(self, vms):
        for vm in vms:
            vm.server_id = self.id
        return {vm.id: vm for vm in vms}

    def spawn_vm(self, vm):
        vm.server_id = self.id
        self.vms[vm.id] = vm

    def spawn_vm_group(self, cpu=[0.5], ram=[0.5]):

        start_id = max(self.vms.keys()) + 1 if self.vms else 0

        assert len(cpu) == len(ram), "cpu and ram lists must match"

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
                return vm.host_task(task)
        return False