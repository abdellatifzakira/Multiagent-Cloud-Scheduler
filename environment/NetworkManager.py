import numpy as np

class NetworkManager:
    def __init__(self, server_farm = None):
        self.server_farm = server_farm
        self.servers = self.server_farm.servers.values() in server_farm if server_farm is not None else None
        
        self.data_packets = {}
        
        self.routes = {}
        
        
        # (parent : sender, child : receiver) : [sent data, time, status: 0 -> pending, 1 -> sent]
        
    
    def resolve_routing(self):
        for key, value in self.data_packets.items() :
            speed = self.server_farm.bandwidths[(min(key[0].server.id,key[1].server.id), max(key[0].server.id,key[1].server.id))]
            self.routes[key] = [key[0].server, key[1].server, value[0], speed]
        
    
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
        sent_data = {}

        # Iterate over a COPY to safely modify original dict
        for duo, value in list(self.routes.items()):

            src, dst, remaining_data, bw = value

            key = (src, dst)
            sent_data.setdefault(key, 0)

            max_transfer = bw * time_step
            

            while remaining_data > 0 and sent_data[key] < max_transfer:

                remaining_bw = max_transfer - sent_data[key]
                transfer = min(remaining_data, remaining_bw)

                # Write data
                dst.received_data[duo] = dst.received_data.get(duo, 0) + transfer

                # Update counters
                sent_data[key] += transfer
                remaining_data -= transfer

            # Update the route's remaining data
            value[2] = remaining_data

            # Remove completed transfers
            if remaining_data <= 0:
                self.routes.pop(duo)
    