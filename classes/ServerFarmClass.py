import igraph as ig

class Server_Farm:
  def __init__(
    self,
    id: int,
    graph: ig.Graph(), # type: ignore
    servers: list,
    num_servers: int,
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
        return {server.id: server}
    
    def available_servers(self):
        return [server for server in self.servers.values() if server.is_available()]
