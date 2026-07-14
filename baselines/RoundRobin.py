import numpy as np

class RoundRobinScheduler:
    def __init__(self,
                 server_farm = None,
                 data_transfer_manager = None,
                 margin = 0.0,
                 job_manager = None,
                 ):
        self.server_farm = server_farm
        self.job_manager = job_manager
        self.jobs = self.job_manager.update_job_list(0)
        self.job_dict = self.populate_job(self.jobs)
        self.margin = margin
        self.data_transfer_manager = data_transfer_manager
        self.servers = self.populate_servers()
        self.ready_tasks = self.find_entry_tasks()
        self.pointer = 0
        self.running_tasks = []

    

    
    def find_entry_tasks(self):
        ready_tasks = []
        for job in self.jobs :
            ready_tasks.extend(job.get_entry_points())
        for task in ready_tasks:
            task.status = 1
        return ready_tasks

    def find_ready_tasks(self, _t):

        # keep old unscheduled tasks
        ready_task = [
            task for task in self.ready_tasks
            if task.status == 1
        ]
        new_jobs = self.job_manager.update_job_list(_t)
        
        # add new jobs to the dictionary 
        for job in new_jobs:
            self.job_dict[job.id] = job
        
        for job in new_jobs:
            self.jobs.append(job)
            ready_task.extend(job.get_entry_points())

        # check active jobs for newly unlocked tasks
        for job in self.jobs:
            for task in job.get_ready_tasks():
                if task not in ready_task:
                    ready_task.append(task)
        
         # sort by arrival time
        ready_task.sort(
            key=lambda task: task.arrival_time
        )
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
    
    def get_finished_jobs(self) :
        _finished_jobs = []
        for _job in self.jobs :
            if {0} == set(tsk.status for tsk in _job.tasks.values()) :
                _finished_jobs.append(_job)
        return _finished_jobs
            
    
    def get_sla_violation_rate(self) :
        
        if self.get_finished_jobs() != []:
            for job in self.get_finished_jobs() :
                job.end_time = max([tsk.end_time for tsk in job.tasks.values()])
            sla_violation_rate = (
                        np.sum([
                            (job.end_time - job.time_arrived) > job.sla_limit
                            for job in self.get_finished_jobs()
                        ])
                        / len(self.get_finished_jobs())
                    ) * 100
            return sla_violation_rate
        else :
            return 0
                    
    def monitor_data_transfer(self):
        data_transfer= 0
        for running in self.running_tasks :
            if len(running.parents)>0 and not running.monitored:
                for parent in running.parents :
                    if parent.server != running.server :
                        data_transfer += self.job_dict[running.job_id].data_transfer_weights[(parent.id,running.id)]
                running.monitored = True
        return data_transfer
        
    def schedule(self):
        n = len(self.servers)
        _t = 0
        time_line = []
        cpu_usage = {s  : [] for s in self.server_farm.servers.values()}
        power_price = []
        data_transfer = []
        sla_violation = []
        workload_std = []
        server_schedules = {s : [] for s in self.server_farm.servers.values()}
        while self.job_manager.workload or len(self.running_tasks)>0 or len(self.ready_tasks)>0:
            for task in self.ready_tasks :

                    server = self.servers[self.pointer]
                    success = server.add_task_to_queue(task)
                    server.execute_tasks(_t)
                    self.pointer = (self.pointer + 1) % n

            for server in self.server_farm.servers.values() :
                server_schedules[server].append(len(list(server.task_queue))) 
                     
            for s in cpu_usage.keys() :
                cpu_usage[s].append(s.cpu_utilization())
            
            
            power_price.append(self.server_farm.get_power_price())
            self.ready_tasks = self.find_ready_tasks(_t = _t)
            self.running_tasks = self.find_running_tasks()
            data_transfer.append(self.monitor_data_transfer())
            sla_violation.append(self.get_sla_violation_rate())
            workload_std.append(np.std([cpu_usage[s][_t] for s in cpu_usage.keys()]))
            time_line.append(_t)
                
            
            self.server_farm.update_farm_state(t=_t)
            _t += 1


        
        
        return  (time_line,
                cpu_usage,
                power_price,
                server_schedules,
                data_transfer,
                sla_violation,
                workload_std
                )


