import gymnasium as gym
import numpy as np
from utilities.JobManager import JobManager
from environment.MetricsManager import MetricsManager
from environment.NetworkManager import NetworkManager
import copy

class CloudEnv(gym.Env):

    def __init__(self,
                 server_farm = None,
                 metrics_manager = None,
                 job_manager = None,
                 agent = None,
                 time_step = 0.01,
                 network_manager = None,
                 network_overhead = False,
                 evaluation = None,
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
        self.agent.build()

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
        self.batch_size = 1
        self.ready_tasks = []
        
        self.results = {}
        self.reward_buffer = []
        
        self.server_farm_backup = copy.deepcopy(server_farm)
        self.jobs =  copy.deepcopy(job_manager.jobs)
        
        self.testing = evaluation
        
        self.mod = 'TRAIN'
    
    def is_running(self) :
        return ( self.job_manager.workload or
            len(self.job_manager.running_tasks)>0 or
            len(self.job_manager.ready_tasks)>0 or
            self.job_manager.pending_tasks > 0 )
        

    
    
    def reset(self, seed=None, options=None, episode = 0):
        super().reset(seed=seed)
        if episode > 0:
            farm = copy.deepcopy(self.server_farm_backup)
            if self.mod == 'TEST':
                jobs = copy.deepcopy(self.testing)
            else:
                jobs = copy.deepcopy(self.jobs)
            self.server_farm = farm
            assert all(tsk.status not in {0,2} for job in jobs for tsk in job.tasks.values())
            self.metrics_manager = MetricsManager(server_farm=farm, jobs=jobs)
            self.job_manager = JobManager(jobs=jobs)
            self.network_manager = NetworkManager(farm)
            self.network_manager.server_farm = farm
        
        self.current_time = 0.0
        self.metrics_manager.initialize()
        self.ready_tasks = self.job_manager.update_ready_tasks()
        self.metrics_manager.set_name(self.agent.name)
        return self.get_state()
    
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
        state = []
        reward = 0
        if self.agent.trainable:
            state, cpu_std = self.get_state()
            reward = - cpu_std
            self.reward_buffer.append(reward)


        return state, reward, done
    
    
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
    
    
    def get_state(self):
        if len(self.ready_tasks) >= self.batch_size:
            task_cpu = self.ready_tasks[:self.batch_size][0].cpu
        else :
            task_cpu = 0
        state = []
        for s in self.server_farm.servers.values():
            current = s.virtual_cpu_efficiency()
        
            max_available_vm_cpu = sum(
                vm.cpu - vm.used_cpu
                for vm in s.vms.values()
            )
            
            queue_pressure = np.sum(
                [tsk.cpu for i, tsk in enumerate(s.task_queue)]
            )
            queue_pressure_norm = max(0, queue_pressure - max_available_vm_cpu)/(s.c_cpu*s.virtualization_level)
            
            score = current + queue_pressure_norm
            state.append(score)
            state.append(s.get_cpu_usage_variation(task_cpu))
        
        try :
            cpu_std = self.metrics_manager.results['CPU_STD'][-1]
            power = self.metrics_manager.results['POWER'][-1]
            mean_completion = self.metrics_manager.results['JOB_MEAN_COMPLETION_TIME'][-1]
        except IndexError:
            cpu_std = 0
            power = 0
            mean_completion = 0
        state.append(cpu_std)
        return state, cpu_std
    
    def is_scheduling_time(self):
        return len(self.ready_tasks) >= self.batch_size
