import numpy as np

class Environment:
    def __init__(self,
                 server_farm = None,
                 metrics_manager = None,
                 job_manager = None,
                 scheduler = None,
                 time_step = 0.01,
                 ):
        self.server_farm = server_farm
        self.job_manager = job_manager
        self.metrics_manager = metrics_manager
        self.scheduler = scheduler
        self.time_step = time_step
        self.scheduler.server_farm = self.server_farm
        self.scheduler.servers = self.server_farm.servers
        
    def is_running(self) :
        return ( self.job_manager.workload or
                len(self.job_manager.running_tasks)>0 or
                len(self.job_manager.ready_tasks)>0 or
                self.job_manager.pending_tasks > 0 )
    
    def reset(self):
        self.server_farm.reset()
        self.job_manager.reset()
        self.metrics_manager.initialize()
    
        
    def run(self):
        t = 0
        
        self.job_manager.initialize(t = t)
        self.metrics_manager.initialize()
        self.metrics_manager.set_name(self.scheduler.name)
        
        while self.is_running() :
            
            self.job_manager.pending_tasks = 0
            
            self.job_manager.update_arrival_jobs(t)

            self.job_manager.update_ready_tasks()
            
            self.scheduler.assign_tasks(self.job_manager.ready_tasks, t)

            for server in self.server_farm.servers.values() :
                server.execute_tasks(t)
            for server in self.server_farm.servers.values() :
                self.job_manager.pending_tasks += len(list(server.task_queue))
                
            self.job_manager.update_running_tasks()
            self.job_manager.update_finished_jobs()
            self.server_farm.update_farm_state(t=t, time_step = self.time_step)
            self.metrics_manager.collect_data(self.job_manager, t)
            
            
            
            
            t += self.time_step
        
        
        print("==================================================")
        print("SUMMARY RUN FOR SCHEDULER : ", self.scheduler.name)
        print("==================================================")
        
        self.metrics_manager.print_experience_summary()
        self.metrics_manager.print_after_run_check()
        self.metrics_manager.print_infrastructure_details()
        

        
            
        return  self.metrics_manager.results


