class RoundRobinScheduler:
    def __init__(self, servers):
        self.server_list = list(servers)
        self.pointer = 0

    def schedule(self, task):
        n = len(self.server_list)
        for _ in range(n):
            server = self.server_list[self.pointer]
            self.pointer = (self.pointer + 1) % n
            success = server.host_task_in_server(task)
            if not success:
                return False  
        return success  