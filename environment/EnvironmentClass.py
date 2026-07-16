import numpy as np

class Environment:
    def __init__(self,
                 server_farm = None,
                 metrics_manager = None,
                 job_manager = None,
                 scheduler = None,
                 ):
        self.server_farm = server_farm
        self.job_manager = job_manager
        self.metrics_manager = metrics_manager
        self.scheduler = scheduler
        self.scheduler.server_farm = server_farm
        self.scheduler.servers = self.server_farm.servers
        
    def is_running(self) :
        return ( self.job_manager.workload or
                len(self.job_manager.running_tasks)>0 or
                len(self.job_manager.ready_tasks)>0 or
                self.job_manager.pending_tasks > 0 )
  

        
    def run(self):
        t = 0
        
        self.job_manager.initialize(t = t)
        self.metrics_manager.initialize()
        
        while self.is_running() :
            
            self.job_manager.pending_tasks = 0
            
            self.job_manager.update_arrival_jobs(t)

            self.job_manager.update_ready_tasks()
            
            self.scheduler.assign_tasks(self.job_manager.ready_tasks, t)

            for server in self.server_farm.servers.values() :
                server.execute_tasks(t)
                self.job_manager.pending_tasks += len(list(server.task_queue))
                
            self.job_manager.update_running_tasks()
            self.job_manager.update_finished_jobs()
            self.server_farm.update_farm_state(t=t)
            self.metrics_manager.collect_data(self.job_manager, t)
            
            t += 1

        return  self.metrics_manager.results


