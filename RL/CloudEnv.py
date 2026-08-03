import gymnasium as gym
import numpy as np
from utilities.ResultPlotter import plot_metrics
import random

class CloudEnv(gym.Env):

    def __init__(self,
                 server_farm = None,
                 metrics_manager = None,
                 job_manager = None,
                 agent = None,
                 time_step = 0.01,
                 network_manager = None,
                 network_overhead = False,
                 ):
        super().__init__()

        self.server_farm = server_farm
        self.job_manager = job_manager
        self.metrics_manager = metrics_manager
        self.network_manager = network_manager
        self.network_overhead = network_overhead
        self.time_step = time_step
        self.agent = agent
        self.current_time = 0.0
        
        self.server_farm.communication_enabled = self.network_overhead 
        self.server_farm.set_communication_mode()
        
        network_manager.server_farm = server_farm

        self.num_servers = len(server_farm.servers)

        self.action_space = gym.spaces.Discrete(self.num_servers)
        self.agent.action_space = self.action_space 

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
        self.batch_size = 1
        self.ready_tasks = []
        
        self.results = {}

    
    
    def is_running(self) :
        return ( self.job_manager.workload or
            len(self.job_manager.running_tasks)>0 or
            len(self.job_manager.ready_tasks)>0 or
            self.job_manager.pending_tasks > 0 )
        

    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_time = 0.0
        self.metrics_manager.initialize()
        self.ready_tasks = self.job_manager.update_ready_tasks()
        self.metrics_manager.set_name("RANDOM")

        return np.array([1.0], dtype=np.float32), {}
    
    def step(self, action):
        
        self.job_manager.pending_tasks = 0

        done = False
        if len(self.ready_tasks) >= self.batch_size:
            task = self.ready_tasks[:self.batch_size][0]
            server = self.server_farm.servers[action]
            if server.first_check(task):
                    server.add_task_to_queue(task, self.current_time)
            
        for server in self.server_farm.servers.values() :
            server.execute_tasks(self.current_time)
        for server in self.server_farm.servers.values() :
            self.job_manager.pending_tasks += len(list(server.task_queue))

        self.job_manager.update_running_tasks()
        self.job_manager.update_finished_jobs()
        self.server_farm.update_farm_state(t=self.current_time, time_step = self.time_step)
        self.metrics_manager.collect_data(self.job_manager, self.current_time)
              
        if self.network_overhead :
            submitted_data = self.server_farm.submit_packets()
            if len(submitted_data) > 0 :
                self.network_manager.update_data_packets(submitted_data)
                self.network_manager.resolve_routing()
            self.network_manager.distribute_data_payloads(self.current_time, self.time_step)
        
        self.current_time += self.time_step

        self.job_manager.update_arrival_jobs(self.current_time)
        
        self.ready_tasks = self.job_manager.update_ready_tasks()

        if not self.is_running():
            done = True

        reward = 0  

        obs = np.array([1.0], dtype=np.float32)

        return obs, reward, done, False, {}
    
    
    def log(self):
        print("==================================================")
        print("SUMMARY RUN FOR SCHEDULER : ")
        print(self.agent.full_name)
        print("==================================================")
        
        self.metrics_manager.print_experience_summary()
        self.metrics_manager.print_after_run_check()
        self.metrics_manager.print_infrastructure_details()
        
        if self.network_overhead :
            self.metrics_manager.print_data_integrity_report()
        else :
            print("[INFO] : the data integrity report is not available when\n"
                  "when the communication overhead is disabled")
        print("==================================================")
        print("END SUMMARY RUN FOR SCHEDULER : ", self.agent.name)
        print("==================================================")
    
    
    
    def get_results(self) :
        return self.metrics_manager.results
    
