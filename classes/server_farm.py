import igraph as ig

class Server_Farm:
  def __init__(
    self,
    id: int,
    graph: ig.Graph(), # type: ignore
    servers: list,
    num_servers: int
  ):
    
    self.id = id
    
    self.graph = graph
    
    self.servers = self.populate_servers(servers)
    
    self.num_servers = num_servers
    
    self.all_cpus = sum([server.c_cpu for server in self.servers.values()])
    
    self.all_rams = sum([server.c_ram for server in self.servers.values()])
    
  @property
  def curr_cpus_util(self):
    return [server.cpu_utilization_rate for server in self.servers.values()]
  
  @property
  def curr_pwrs(self):
    return [server.total_power for server in self.servers.values()]
  
  @property
  def get_price(self):
    total_power = round(sum(server.total_power for server in self.servers.values()), 2)
    # Threshold from "Impact of dynamic energy pricing schemes
    # on a novel multi-user home energy management system".
    threshold = 1.5
    # Price from "Optimal residential load control with price
    # prediction in real-time electricity pricing environments".
    real_time_pricing_low = 5.91
    real_time_pricing_high = 8.27
    if total_power <= threshold:
      price = total_power * real_time_pricing_low
    else:
      price = total_power * real_time_pricing_high
    return round(price, 2)
  
  def populate_servers(self, servers):
    return {server.id: server for server in servers}

# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from server import Server
    from vm import Vm

    # Create VMs for each server
    vms_server1 = [Vm(id=1, cpu=2.0, ram=4.0), Vm(id=2, cpu=1.5, ram=3.0)]
    vms_server2 = [Vm(id=3, cpu=3.0, ram=6.0), Vm(id=4, cpu=2.5, ram=5.0)]

    # Create servers
    server1 = Server(id=1, server_farm_id=1, vms=vms_server1, c_cpu=4.0, c_ram=8.0, alpha=0.5, beta=0.3)
    server2 = Server(id=2, server_farm_id=1, vms=vms_server2, c_cpu=5.5, c_ram=11.0, alpha=0.6, beta=0.4)
    # Create a graph for the server farm
    graph = ig.Graph(directed=True)
    graph.add_vertices(2)  # Two servers
    graph.add_edges([(0, 1)])  # Server 1 connects to Server 2

    # Create the server farm
    server_farm = Server_Farm(id=1, graph=graph, servers=[server1, server2], num_servers=2)

    # Print server farm details
    print(f"Server Farm ID: {server_farm.id}")
    print(f"Total CPUs: {server_farm.all_cpus}")
    print(f"Total RAMs: {server_farm.all_rams}")
    print(f"Current CPU Utilization Rates: {server_farm.curr_cpus_util}")
    print(f"Current Power Consumption: {server_farm.curr_pwrs}")
    print(f"Current Price: {server_farm.get_price}")
