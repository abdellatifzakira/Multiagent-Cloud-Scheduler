class RoundRobinScheduler:
    def __init__(self,
                 server_farm = None,
                 jobs : list = None,
                 data_transfer_manager = None,
                 horizon = None,
                 ):
        self.server_farm = server_farm
        self.jobs = jobs
        self.data_transfer_manager = data_transfer_manager
        self.servers = self.populate_servers()
        
        
    def populate_farms(self):
        return self.server_farm.servers
        

    def schedule(self, task):
        n = len(self.server_list)
        for _ in range(n):
            server = self.server_list[self.pointer]
            self.pointer = (self.pointer + 1) % n
            success = server.host_task_in_server(task)
            if not success:
                return False  
        return success  

    
    
    
    
    
    def schedule(self):
        return