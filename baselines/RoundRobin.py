import numpy as np

class RoundRobinScheduler:
    def __init__(self,
                 server_farm = None,
                 jobs : list = None,
                 data_transfer_manager = None,
                 margin = 0.1,
                 ):
        self.server_farm = server_farm
        self.jobs = jobs
        self.job_dict = self.populate_job(jobs)
        self.margin = margin
        self.data_transfer_manager = data_transfer_manager
        self.servers = self.populate_servers()
        self.ready_tasks = self.find_entry_tasks()
        self.horizon = self.compute_simulation_horizon()
        self.pointer = 0
        self.running_tasks = []
    def compute_simulation_horizon(self):
        horizon = np.sum(job.get_deadline() for job in self.jobs)
        return horizon*(1 + self.margin)
    
    def find_entry_tasks(self):
        ready_tasks = []
        for job in self.jobs :
            ready_tasks.extend(job.get_entry_points())
        for task in ready_tasks:
            task.status = 1
        return ready_tasks

    def find_ready_tasks(self):
        ready_task = []
        for job in self.jobs :
            ready_task.extend(job.get_ready_tasks())
        return ready_task
    
    def find_running_tasks(self):
        ready_task = []
        for job in self.jobs :
            ready_task.extend(job.get_running_tasks())
        return ready_task
        
    def populate_servers(self):
        return self.server_farm.servers
    
    def populate_job(self, jobs) :
        return {job.id : job for job in jobs}
    
    def monitor_data_transfer(self):
        data_transfer= 0
        for running in self.running_tasks :
            if len(running.parents)>0 and not running.monitored:
                for parent in running.parents :
                    if parent.server_id != running.server_id :
                        data_transfer += self.job_dict[running.job_id].data_transfer_weights[(parent.id,running.id)]
                running.monitored = True
        return data_transfer
        
    def schedule(self):
        n = len(self.servers)
        _t = 0
        time_line = []
        task_state = {
                        task_key: []
                        for job in self.jobs
                        for task_key in job.tasks.values()
                    }
        cpu_usage = {s  : [] for s in self.server_farm.servers.values()}
        power_price = []
        data_transfer = []
        server_schedules = {_id : [] for _id in range(n)}
        while _t < self.horizon :
            for task in self.ready_tasks :
                    server = self.servers[self.pointer]
                    success = server.host_task_in_server(task)
                    if success:
                        server_schedules[self.pointer].append(_t) 
                    self.pointer = (self.pointer + 1) % n
                     
            for s in cpu_usage.keys() :
                cpu_usage[s].append(s.cpu_utilization())
            
            for tsk in [_task for _job in self.jobs for _task in _job.tasks.values()] :
                task_state[tsk].append(tsk.status)
            
            power_price.append(self.server_farm.get_power_price())
            self.ready_tasks = self.find_ready_tasks()
            self.server_farm.update_farm_state()
            self.running_tasks = self.find_running_tasks()
            data_transfer.append(self.monitor_data_transfer())
            time_line.append(_t)
            _t += 1
            
        return  time_line, task_state, cpu_usage, power_price, server_schedules, data_transfer


