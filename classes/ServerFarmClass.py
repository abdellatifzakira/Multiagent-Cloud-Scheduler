import igraph as ig
import numpy as np
import random
from collections import deque
try :
    from ServerClass import Server
except ModuleNotFoundError :
    from classes.ServerClass import Server
class Server_Farm:
    _farm_counter =  0
    def __init__(
    self,
    id: int = None,
    servers: list = None,
    num_servers: int = None,
    graph: ig.Graph() = None, # type: ignore
    power_price: float = 0.1
  ):
        if id is None:
            self.id = Server_Farm._farm_counter
            Server_Farm._farm_counter += 1
        else:
            self.id = id
            Server_Farm._farm_counter = max(Server_Farm._farm_counter, id + 1)
        
        self.graph = graph
        
        self.servers = self.populate_servers(servers) # create a dict of servers with their ids as keys for faster access
        
        self.num_servers = num_servers
        
        self.power_price = power_price
        
        self.all_cpus = sum([server.c_cpu for server in self.servers.values()]) if servers else 0
        
        self.all_rams = sum([server.c_ram for server in self.servers.values()])  if servers else 0
        
        self.communication_speed = []
    
    def spawn_server(self, server):
        server.server_farm_id = self.id
        self.servers[server.id] = server
        
    def populate_servers(self, servers):
        if servers is None :
            return None
        for server in servers:
            server.server_farm_id = self.id
        return {server.id: server for server in servers}
    
    def available_servers(self):
        return [server for server in self.servers.values() if server.is_available()]
    
    
    def get_power_price(self):
        return sum(server.get_power_consumption() for server in self.servers.values())*self.power_price
    
    # Naive first in list first served at first to be enhanced later on
    def host_task_in_farm(self, task):
        for server in self.available_servers():
            if server.host_task_in_server(task) :
                return True
        return False

    def update_farm_state(self,t, time_step = 1):
        for server in self.servers.values() :
            server.time_step_vm(_t=t,time_step = time_step)
        
        
    
    @staticmethod
    def build_random_server_farms(
                                cpu_range = [512, 2048],
                                ram_range = [512, 4096],
                                storage_range = [512, 16384],
                                max_vms_count = 3,
                                alphas = [32, 128],
                                betas = [2, 4],
                                server_count = 3,
                                power_price = 0.1,
                                virtual_allocation = [0.85, 0.9]
                                ):
        
        assert server_count > 0, \
            "Invalid simulation parameter : server count must be > 1"
        
        assert min(cpu_range) > 100 and min(ram_range) > 100, \
            "Invalid simulation parameters: CPU and RAM minimum values must exceed 100."
        
        assert max_vms_count >= 2, \
            "Invalid simulation parameters: vm count must be at least 2"
        assert min(cpu_range) / max_vms_count >= 16, \
            f"Invalid simulation parameters: each VM may receive only {int(min(cpu_range) / max_vms_count)} CPU units. " \
            "For reliable simulation results, ensure at least 16 CPU units per VM.\n" \
            "Consider reducing max_vms_count or increasing the minimum CPU value in cpu_range."

        assert min(ram_range) / max_vms_count >= 128, \
            f"Invalid simulation parameters: each VM may receive only {int(min(ram_range) / max_vms_count)} RAM units. " \
            "For reliable simulation results, ensure at least 128 RAM units per VM.\n" \
            "Consider reducing max_vms_count or increasing the minimum RAM value in ram_range."
        
        c_cpu = [round(np.random.uniform(low= min(cpu_range), high=max(cpu_range)),ndigits = 0) for _ in range(server_count)]
        c_ram = [round(np.random.uniform(low= min(ram_range), high=max(ram_range)),ndigits = 0) for _ in range(server_count)]
        c_storage = [round(np.random.uniform(low= min(storage_range), high=max(storage_range)),ndigits = 0) for _ in range(server_count)]
        alpha = [round(np.random.uniform(low= min(alphas), high=max(alphas)),ndigits = 2) for _ in range(server_count)]
        beta = [round(np.random.uniform(low= min(betas), high=max(betas)),ndigits = 2) for _ in range(server_count)]
        server_list = []
        for _ in range(server_count):

            server = Server(
                c_cpu=c_cpu[_],
                c_ram=c_ram[_],
                storage=c_storage[_],
                alpha=alpha[_],
                beta=beta[_]
            )
            
            virtual_quota = round(np.random.uniform(low= min(virtual_allocation), high=max(virtual_allocation)),ndigits = 3)

            vm_count = random.choice(range(2, max_vms_count + 1))

            vm_cpu = []
            vm_ram = []

            remaining_cpu = c_cpu[_]*virtual_quota
            remaining_ram = c_ram[_]*virtual_quota

            min_cpu = 32
            min_ram = 128

            for i in range(vm_count):

                remaining_vms = vm_count - i

                if remaining_vms == 1:
                    cpu = remaining_cpu
                    ram = remaining_ram

                else:
                    max_cpu = remaining_cpu - (remaining_vms - 1) * min_cpu
                    max_ram = remaining_ram - (remaining_vms - 1) * min_ram

                    cpu = round(np.random.uniform(min_cpu, max_cpu))
                    ram = round(np.random.uniform(min_ram, max_ram))

                vm_cpu.append(cpu)
                vm_ram.append(ram)

                remaining_cpu -= cpu
                remaining_ram -= ram


            server.spawn_vm_group(
                cpu=vm_cpu,
                ram=vm_ram
            )

            server_list.append(server)
        farm = Server_Farm(
            servers = server_list,
            power_price = power_price,
            num_servers = server_count
        )
        return farm
                    
    
    def reset(self):
        for server in self.servers.values() :
            self.hosted_tasks = {}
            self.task_queue = deque()
            for vm in server.vms.values():
                vm.hosted_task = {}
                vm.pending_tasks = deque()
                vm.used_cpu = 0.0
                vm.used_ram = 0.0
                
                


# QUICK TESTS :
if __name__ == "__main__":
    from ServerClass import Server
    from TaskClass import  Task
    
    server_1 = Server(
        c_cpu= 1.0,
        c_ram=1.0,
        server_farm_id=0,
        alpha=0.1
    )
    
    server_2 = Server(
        c_cpu= 1.0,
        c_ram=1.0,
        server_farm_id=0,
        alpha=0.1
    )
    
    farm = Server_Farm(
        id = 0,
        servers= [server_1, server_2],
        num_servers= 2
    )
    
    server_1.spawn_vm_group(cpu = [0.3, 0.1], ram=[0.25, 0.25])
    server_2.spawn_vm_group(cpu = [0.3 , 0.3, 0.1], ram=[0.25, 0.25, 0.25])
    
    task_2 = Task(
        id = 0,
        job_id= 0,
        cpu=0.1,
        ram=0.1,
        status=3,
        runtime=20
    )
    
    task_1 = Task(
        id = 1,
        job_id= 0,
        cpu=0.1,
        ram=0.1,
        status=3,
        runtime=20
    )
    
    server_1.host_task_in_server(task = task_1)
    server_2.host_task_in_server(task = task_2)
    
    print(len(farm.available_servers())) # expecting 2 both servers still have computational power
    print("SERVER 2 IS AVAILABLE"  if server_2.is_available() else "SERVER 2 IS UNAVAILABLE" ) # AVAILABLE
    print("SERVER 1 IS AVAILABLE"  if server_1.is_available() else "SERVER 1 IS UNAVAILABLE" ) # AVAILABLE
    
    # overloading the first server :
    
    task_3 = Task(
        id = 2,
        job_id= 0,
        cpu=0.1,
        ram=0.1,
        status=3,
        runtime=20
    )
    
    server_1.host_task_in_server(task = task_3)
    
    print(len(farm.available_servers())) # expecting 1, only one server still have computational power
    print("SERVER 2 IS AVAILABLE"  if server_2.is_available() else "SERVER 2 IS UNAVAILABLE" ) # AVAILABLE
    print("SERVER 1 IS AVAILABLE"  if server_1.is_available() else "SERVER 1 IS UNAVAILABLE" ) # UNAVAILABLE
    
    computedPrice = farm.power_price*(server_1.cpu_utilization()*server_1.alpha +
                                      server_1.static_power +
                                      server_2.static_power +
                                      server_2.cpu_utilization()*server_2.alpha)
    print(f"EXPECTED POWER PRICE : ", round(computedPrice,3) ,"COMPUTED POWER PRICE : ", round(farm.get_power_price(),3))
    
    
    