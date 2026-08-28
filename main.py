from utilities.DAG_handlers import *
from classes.JobClass import generate_workload
from baselines.RoundRobin import RoundRobinScheduler
from comparison.LeastLoaded import LeastLoadedScheduler
from comparison.LeastCommunication import DataLocalityAwareScheduler
from comparison.LeastEnergy import EnergyAwareScheduler
from utilities.helpers import *
from ExperimentRunner import Experiment

import time

from RL.agents.RandomAgent import RandomAgent
from RL.agents.SequentialDQNAgent import SequentialDQNAgent
from RL.agents.DQNAgent import DQNAgent
from classes.ServerFarmClass import build_random_server_farms



infra_seed = 13
training_seed = 24
testing_seed = 1
schedulers_seed = 42

start_time = time.time()
jobs, _ = generate_workload(
        seed=training_seed,
        num_jobs=[100, 50, 25],
        mean_job_gap=[0.1, 0.05, 0.01],
        num_tasks_per_job=5,
        cpu_req_per_task=[12, 64],
        ram_req_per_task=[2, 32],
        task_size=[32, 64],
        edge_probability=0.0,
        instructions_per_task=[
            250e5,
            500e5,
        ],
        data_transfer_range=[
            512,
            1024,
        ],
    )
    
    
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



        

plot_server_network(
            farm=server_farm
        )
jobs_test, _ = generate_workload(
                seed=testing_seed,
                num_jobs=[100, 50, 25],
                mean_job_gap=[0.1, 0.05, 0.01],
                num_tasks_per_job=5,
                cpu_req_per_task=[12, 64],
                ram_req_per_task=[2, 32],
                task_size=[32, 64],
                edge_probability=0.0,
                instructions_per_task=[
                    250e5,
                    500e5,
                ],
                data_transfer_range=[
                    512,
                    1024,
                ],
            )
plot_job_dags(
           jobs_list=jobs_test
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
            jobs=jobs,
            scenarios_edges=[],
            schedulers=[
                DQNAgent(
                    epsilon=1.0,
                    epsilon_decay=0.95,
                    seed=schedulers_seed,
                    buffer_capacity=10_000),
                SequentialDQNAgent(
                            epsilon=1.0,
                            epsilon_decay=0.95,
                            seed=schedulers_seed,
                            buffer_capacity=10_000),
                #LeastLoadedScheduler(mode='CPU'),
                #RoundRobinScheduler(),
                #DataLocalityAwareScheduler(mode='NAIVE'),
                #DataLocalityAwareScheduler(mode='BALANCED'),
                #EnergyAwareScheduler(),
                RandomAgent(seed=schedulers_seed)
                
            ],
            time_step=0.005,
            network_overhead_enabled=True,
            power_model="DEFAULT",
            evaluation=jobs_test,
            episodes=500,
            batch_size=25,
            clamp_results = True
        )


exp.build_environment()
exp.run_experiment()

print("=" * 50)
print(
    "EXPERIMENT RAN FOR : "
    f"{time.time() - start_time:.2f} s"
)
exp.plot_results()