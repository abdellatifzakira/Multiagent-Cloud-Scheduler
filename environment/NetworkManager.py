import numpy as np

class NetworkManager:
    def __init__(self, server_farm = None):
        self.server_farm = server_farm
        self.servers = self.server_farm.servers.values() in server_farm if server_farm is not None else None
        
        self.data_packets = {}
        
        self.routes = {}
        
        
        # (sender, receiver, time) : [sent data, status: 0 -> pending, 1 -> sent]
        
    
    def resolve_routing(self):
        for key, value in self.data_packets.items() :
            speed = self.server_farm.bandwidths[(min(key[0].server.id,key[1].server.id), max(key[0].server.id,key[1].server.id))]
            self.routes[key] = [key[0].server, key[1].server, value[0], speed]
        
    
    def update_data_packets(self, new_packets):
        for packets in new_packets :
            for key, value in packets.items():
                assert value[1] == 1, f"[Fatal Error] : Unsend data was submitted"
                self.data_packets[key] = value
        
        self.data_packets = {k: v for k, v in sorted(self.data_packets.items(), key=lambda item: item[0][2])}
        print(f"{new_packets = }")
    
    
    def resolve_data_transfer(self, t, time_step):
        ...
    
    
    def distribute_data_payloads(self):
        ...
    
    
try:
    ...
except KeyError:
    ...