from utilities.DAG_handlers import *
from classes.JobClass import generate_workload
from ExperimentRunner import Experiment
from baselines.RoundRobin import RoundRobinScheduler
from RL.agents.Agent import Agent

import time
from classes.ServerFarmClass import build_random_server_farms

class RoundRobinTwin(Agent):
    def __init__(self, seed = 123):
        super().__init__('RRA', 'ROUND ROBIN AGENT')
        self.trainable = False
        self.seed = seed
        self.index = 0
        self.batch_size = 4
        self.num_servers = 4
        
    
    def take_action(self):
        action  = [0 for _ in range(self.batch_size)]
        for _ in range(self.batch_size):
            action[_] = self.index
            self.index = (self.index + 1)%self.num_servers
        return action

    
    
    def observe(self, state):
        tasks = state["TASKS"]
        self.batch_size = len(list(tasks.keys()))
        

    def build(self):
        self.action_space.seed(self.seed)
    
    def update(self, *args):
        pass
    



infra_seed = 13
training_seed = 24
testing_seed = 1
schedulers_seed = 42

num_jobs = 100
num_tasks_per_job = 5 # 100 task in total

num_servers = 4

start_time = time.time()
server_farm = build_random_server_farms(
        cpu_range=[1024, 1024 * 2],
        ram_range=[4096, 4096 * 4],
        storage_range=[20480, 20480 * 2],
        compute_power_range=[
            1e9,
            2e9,
        ],
        max_vms_count=5,
        alphas=[100, 500],
        betas=[2, 5],
        server_count=4,
        virtual_allocation=[
            0.9,
            0.95,
        ],
        bandwidth=[
            4096.0,
            16384.0,
        ],
        mode="ROUNDROBIN",
        seed=infra_seed,
    )



        


jobs_test, _ = generate_workload(
                seed=testing_seed,
                num_jobs=[num_jobs],
                mean_job_gap=[0.0],
                num_tasks_per_job=num_tasks_per_job,
                cpu_req_per_task=[12, 64],
                ram_req_per_task=[2, 32],
                task_size=[32, 64],
                edge_probability=0.25,
                instructions_per_task=[
                    250e5,
                    500e5,
                ],
                data_transfer_range=[
                    512,
                    1024,
                ],
            )
print("=" * 51)
print("BEFORE RUN CHECK")
print(
           f"ALL TASKS "
           f"{len([task for job in jobs_test for task in job.tasks.values()])}")
print(
           "TASK STATES "
           f"{set(task.status for job in jobs_test for task in job.tasks.values())} "
           " : EXPECTED {1, 3}"
        )
        
print("=" * 51)

exp = Experiment(
            infrastructure=server_farm,
            jobs=jobs_test,
            scenarios_edges=[],
            schedulers=[
                RoundRobinScheduler(mode="CHECK"),
                RoundRobinTwin()
                
            ],
            time_step=0.005,
            network_overhead_enabled= True,
            power_model="DEFAULT",
            evaluation=jobs_test,
            episodes=600,
            batch_size=24,
            clamp_results = False,
            verbose = True
        )


exp.build_environment()
exp.run_experiment()

print("=" * 50)
print(
    "EXPERIMENT RAN FOR : "
    f"{time.time() - start_time:.2f} s"
)
exp.plot_results()