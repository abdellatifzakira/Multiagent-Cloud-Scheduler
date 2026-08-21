# main.py

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
from RL.agents.DQNAgent import DQNAgent
from classes.ServerFarmClass import (
    build_random_server_farms,
)


global_seed = 256
test_seed = 123


jobs, _ = generate_workload(
    seed=global_seed,
    num_jobs=[100, 100, 50],
    mean_job_gap=[0.075, 0.05, 0.01],
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


jobs_test, scenarios = generate_workload(
    seed=test_seed,
    num_jobs=[100, 100, 50],
    mean_job_gap=[0.075, 0.05, 0.01],
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
    f"{len([task for job in jobs for task in job.tasks.values()])}"
)

print(
    "TASK STATES "
    f"{set(task.status for job in jobs for task in job.tasks.values())} "
    " : EXPECTED {1, 3}"
)

print("=" * 51)


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
    seed=global_seed,
)


plot_server_network(
    farm=server_farm
)


exp = Experiment(
    infrastructure=server_farm,
    jobs=jobs,
    scenarios_edges=scenarios,
    schedulers=[
        DQNAgent(
            epsilon=0.05,
            epsilon_decay=0.95,
            seed=global_seed),
        LeastLoadedScheduler(mode='CPU'),
        RoundRobinScheduler(),
        #DataLocalityAwareScheduler(mode='BALANCED')
        RandomAgent(seed=global_seed)
        
    ],
    time_step=0.005,
    network_overhead_enabled=True,
    power_model="DEFAULT",
    evaluation=jobs_test,
    episodes=50,
    batch_size=4,
    clamp_results = True
)


exp.build_environment()

start_time = time.time()

exp.run_experiment()

print("=" * 50)
print(
    "EXPERIMENT RAN FOR : "
    f"{time.time() - start_time:.2f} s"
)

exp.plot_results()