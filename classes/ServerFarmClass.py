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
        
        self.bandwidths = {}
        
        self.communication_enabled = None
        self.power_model = 'DEFAULT'
        
        
        
    def set_communication_mode(self):
        for server in self.servers.values():
            server.network_enabled = self.communication_enabled
    
    def spawn_server(self, server):
        server.server_farm_id = self.id
        self.servers[server.id] = server
        
    def populate_servers(self, servers):
        if servers is None :
            return None
        for server in servers:
            server.server_farm_id = self.id
        return {server.id: server for server in servers}
    
    
    def get_power_price(self):
        return sum(server.get_power_consumption() for server in self.servers.values())*self.power_price
    
    def get_power(self):
            return sum(server.get_power_consumption() for server in self.servers.values())
    


    def update_farm_state(self,t, time_step):
        for server in self.servers.values() :
            server.time_step_vm(_t=t,time_step = time_step)
            
            
    def build_graph(self, bandwidths):
        if not len(bandwidths):
            return None
        edges = list(bandwidths.keys())
        weights = list(bandwidths.values())
        vertices = set()
        
        servers = self.servers.values()
        
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)

        assert len(vertices) <= len(servers), "servers bandwidths mismatch!"

        graph = ig.Graph(directed=False)

        graph.add_vertices(len(servers))
        graph.add_edges(edges)

        # assign weights
        graph.es["bandwidth"] = weights
        
        # assign attributes
        cpus = [s.c_cpu for s in servers]
        rams = [s.c_ram for s in servers]
        graph.vs["cpu"] = cpus
        graph.vs["ram"] = rams
        

        return graph
            


    
    @staticmethod
    def build_random_server_farms(
                                cpu_range = [512, 2048],
                                ram_range = [512, 4096],
                                storage_range = [512, 16384],
                                compute_power_range = [1e9, 5e9],
                                max_vms_count = 3,
                                alphas = [32, 128],
                                betas = [2, 4],
                                server_count = 3,
                                power_price = 0.1,
                                virtual_allocation = [0.85, 0.9],
                                bandwidth = [1024*4, 4096*4],
                                mode = 'ROUNDROBIN'
                                ):
        
        assert mode in ['ROUNDROBIN', 'LEAST_LOADED', 'EXECUTION_TIME', 'COLLABORATIVE'], \
            "[BAD INPUT] : AVAILABLE MODES ROUNDROBIN | LEAST_LOADED | EXECUTION_TIME | COLLABORATIVE"
        
        assert server_count > 0, \
            "Invalid simulation parameter : server count must be > 1"
        
        assert min(cpu_range) > 100 and min(ram_range) > 100, \
            "Invalid simulation parameters: CPU and RAM minimum values must exceed 100."
        
        assert max_vms_count >= 2, \
            "Invalid simulation parameters: vm count must be at least 2"
        assert min(cpu_range) / max_vms_count >= 32, \
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
        c_compute_power = [round(np.random.uniform(low= min(compute_power_range), high=max(compute_power_range)),ndigits = 0) for _ in range(server_count)]
        alpha = [round(np.random.uniform(low= min(alphas), high=max(alphas)),ndigits = 2) for _ in range(server_count)]
        beta = [round(np.random.uniform(low= min(betas), high=max(betas)),ndigits = 2) for _ in range(server_count)]
        #beta = [1 for _ in range(server_count)] # linear power consumption across all servers
        server_list = []
        for _ in range(server_count):
            
            virtual_quota = round(np.random.uniform(low= min(virtual_allocation), high=max(virtual_allocation)),ndigits = 3)

            server = Server(
                id= _,
                c_cpu=c_cpu[_],
                c_ram=c_ram[_],
                storage=c_storage[_],
                compute_power=c_compute_power[_],
                alpha=alpha[_],
                beta=beta[_],
                virtualization_level = virtual_quota,
                mode = mode
            )
            


            vm_count = random.choice(range(2, max_vms_count + 1))

            vm_cpu = []
            vm_ram = []
            vm_compute = []

            remaining_cpu = c_cpu[_]*virtual_quota
            remaining_ram = c_ram[_]*virtual_quota
            remaining_compute = c_compute_power[_]*virtual_quota

            min_cpu = 32
            min_ram = 128
            min_compute = 25e6

            for i in range(vm_count):

                remaining_vms = vm_count - i

                if remaining_vms == 1:
                    cpu = remaining_cpu
                    ram = remaining_ram
                    compute = remaining_compute

                else:
                    max_cpu = remaining_cpu - (remaining_vms - 1) * min_cpu
                    max_ram = remaining_ram - (remaining_vms - 1) * min_ram
                    max_compute = remaining_compute - (remaining_vms - 1) * min_compute
                    
                    cpu = round(np.random.uniform(min_cpu, max_cpu))
                    ram = round(np.random.uniform(min_ram, max_ram))
                    compute = round(np.random.uniform(min_compute, max_compute))
                vm_cpu.append(cpu)
                vm_ram.append(ram)
                vm_compute.append(compute)

                remaining_cpu -= cpu
                remaining_ram -= ram
                remaining_compute -= compute


            server.spawn_vm_group(
                cpu=vm_cpu,
                ram=vm_ram,
                compute=vm_compute
            )
            

            server_list.append(server)
        
        
        bandwidths = {}
        for i in range(len(server_list)):
            for j in range(i+1, len(server_list)):
                    weight = np.random.randint(min(bandwidth), max(bandwidth))
                    bandwidths[(i, j)] = weight
        
        
        farm = Server_Farm(
            servers = server_list,
            power_price = power_price,
            num_servers = server_count
        )
        
        farm.bandwidths = bandwidths
        
        farm.graph = farm.build_graph(bandwidths = bandwidths)
        return farm
    
    ##################################
    def set_power_model(self, model):
        self.power_model = model
        for server in self.servers.values():
            server.set_power_model(model)
    ###################################                
    
    
    def submit_packets(self):
        packets = []
        for server in self.servers.values():
            if server.has_data() :
                packets.append(server.send_data())
        return packets
                
                
