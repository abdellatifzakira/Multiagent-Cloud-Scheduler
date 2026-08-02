import numpy as np

class NetworkManager:
    def __init__(self, server_farm = None):
        self.server_farm = server_farm
        self.servers = self.server_farm.servers.values() if server_farm is not None else None
        
        self.data_packets = {}
        
        self.routes = {}
        
        
        self.bandwidth = {}
        
        # (parent : sender, child : receiver) : [sent data, time, status: 0 -> pending, 1 -> sent]
        
    
    def resolve_routing(self):
        for key, value in self.data_packets.items() :
            speed = self.server_farm.bandwidths[(min(key[0].server.id,key[1].server.id), max(key[0].server.id,key[1].server.id))]
            self.routes[key] = [key[0].server, key[1].server, value[0], speed]
            self.bandwidth[frozenset([key[0].server, key[1].server])] = speed
            self.data_packets[key][2] = 2 # processed
        self.data_packets = {key : self.data_packets[key] for key in self.data_packets.keys() if self.data_packets[key][2] == 1}
        
    
    def update_data_packets(self, new_packets):
        for packets in new_packets :
            for key, value in packets.items():
                assert value[2] == 1, f"[Fatal Error] : Unsend data was submitted"
                self.data_packets[key] = value
        
        # ascending order for chronogical consitency
        self.data_packets = {k: v for k, v in sorted(self.data_packets.items(), key=lambda item: item[1][1])}
        
    def distribute_data_payloads(self, t, time_step):
        if not self.routes:
            return
        effective_bandwidth = {key : value*time_step for key, value in self.bandwidth.items()}
        bandwidth_consumption = effective_bandwidth   
        # each route is the form {(parent, child) : [src, dst, data, bw]}
        for duo in self.routes.keys():
            key = frozenset([self.routes[duo][0], self.routes[duo][1]])
            while self.routes[duo][2] > 0 and bandwidth_consumption[key] > 0:
                transfer = min(self.routes[duo][2], effective_bandwidth[key])
                self.routes[duo][1].receive_data(duo, transfer)
                bandwidth_consumption[key] -= transfer
                self.routes[duo][2] -= transfer

        self.routes = {duo: data for duo, data in self.routes.items() if data[2] > 0}

        