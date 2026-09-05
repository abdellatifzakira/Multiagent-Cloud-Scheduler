import numpy as np

class NetworkManager:
    def __init__(self, server_farm = None):
        self.server_farm = server_farm
        self.servers = self.server_farm.servers.values() if server_farm is not None else None
        
        self.data_packets = {}
        
        self.routes = {}
        
        self.bandwidth = {}
        
        for s in server_farm.servers.values():
            for server in server_farm.servers.values() :
                if server.id != s.id :
                    key_direct = (min(server.id,s.id), max(server.id,s.id))
                    key_inverse = (max(server.id,s.id), min(server.id,s.id))
                    
                    # bw consitency
                    bw = self.server_farm.bandwidths[key_direct]
                    self.bandwidth[key_direct] = bw
                    self.bandwidth[key_inverse] = bw
                    
                    # packets storage
                    self.routes[key_direct] = []
                    self.routes[key_inverse] = []       
                    
        
        # data packet (parent : sender, child : receiver) : [sent data, time, status: 0 -> pending, 1 -> sent, 2 -> processed]
        # route {(server pair) : []}   
    
    def resolve_routing(self):
        for key, value in self.data_packets.items() :
            parent = key[0]
            child = key[1]
            direction = (parent.server.id, child.server.id)
            
            route =self.routes[direction]
            
            route.append({key : value.copy()})
            
            self.data_packets[key][2] = 2 # processed
        self.data_packets = {key : self.data_packets[key] for key in self.data_packets.keys() if self.data_packets[key][2] == 1}
        
    
    def update_data_packets(self, new_packets):
        for packets in new_packets :
            for key, value in packets.items():
                assert value[2] == 1, f"[Fatal Error] : Unsent data was submitted"
                self.data_packets[key] = value
        
        # ascending order for chronogical consitency
        self.data_packets = {k: v for k, v in sorted(self.data_packets.items(), key=lambda item: item[1][1])}
        
    def distribute_data_payloads(self, t, time_step):
        if not self.routes:
            raise ValueError("Routes was not properly initialized !")

        effective_bandwidth = {key : value*time_step for key, value in self.bandwidth.items()}
        # each route is the form {(parent, child) : [data, time, status]}
        for channel in self.routes.keys(): # channel : server 1 -> server 2
            data = self.routes[channel]
            if data:
                competition = len(data)
                assert competition > 0
                fair_share = round(effective_bandwidth[channel]/competition, ndigits= 3)
                for tr in data :
                    for key, transmission in tr.items():
                        transfer = min(transmission[0], fair_share)
                        key[1].server.receive_data(key, transfer)
                        transmission[0] -= transfer

                data = [{key : value} for tr in data for key, value in tr.items()  if value[0] > 0]
                self.routes[channel] = data

        