import igraph as ig

class Server_Farm:
    def __init__(
    self,
    id: int,
    servers: list,
    num_servers: int,
    graph: ig.Graph() = None, # type: ignore
    power_price: float = 0.1
  ):
        self.id = id
        
        self.graph = graph
        
        self.servers = self.populate_servers(servers) # create a dict of servers with their ids as keys for faster access
        
        self.num_servers = num_servers
        
        self.power_price = power_price
        
        self.all_cpus = sum([server.c_cpu for server in self.servers.values()])
        
        self.all_rams = sum([server.c_ram for server in self.servers.values()])
        
        self.communication_speed = []
    
    def spawn_server(self, server):
        server.server_farm_id = self.id
        self.servers[server.id] = server
        
    def populate_servers(self, servers):
        for server in servers:
            server.server_farm_id = self.id
        return {server.id: server for server in servers}
    
    def available_servers(self):
        return [server for server in self.servers.values() if server.is_available()]
    
    
    def get_power_price(self):
        return sum(server.get_power_consumption() for server in self.servers.values())*self.power_price




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
    
    
    