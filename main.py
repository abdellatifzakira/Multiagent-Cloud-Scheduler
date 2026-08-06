from classes.JobClass import Job
from classes.ServerFarmClass import Server_Farm
from utilities.DAG_handlers import *
from baselines.RoundRobin import RoundRobinScheduler
from comparison.LeastLoaded import LeastLoadedScheduler
from comparison.LeastCommunication import DataLocalityAwareScheduler
from comparison.LeastEnergy import EnergyAwareScheduler
import numpy as np
import random
from utilities.helpers import *
from ExperimentRunner import Experiment
from environment.NetworkManager import NetworkManager
from RL.agents.RandomAgent import RandomAgent
from RL.agents.DQNAgent import DQNAgent 
import time

# REPRODUCIBILITY
global_seed = 123
random.seed(global_seed)
np.random.seed(global_seed)

#WORKLOAD
num_jobs = 450
mean_job_gap = 0.01
num_tasks_per_job = 5
jobs_per_phase = num_jobs // 3
edge_probability =  0.25 # controls how fuzzy the jobs are

# Light
light_gap = np.array([round( t , ndigits = 5) for t in np.random.exponential(mean_job_gap*10, jobs_per_phase)])
arrival_light = np.cumsum(light_gap)
# Medium
medium_gap = np.array([round( t , ndigits = 5) for t in np.random.exponential(mean_job_gap*4, jobs_per_phase)])
arrival_medium = np.cumsum(medium_gap) + arrival_light[-1] 
# Surge
surge_gap = np.array([round( t , ndigits = 5) for t in np.random.exponential(mean_job_gap, jobs_per_phase)])
arrival_surge = np.cumsum(surge_gap) + arrival_medium[-1] 

arrival_times = np.concatenate([
    arrival_light,
    arrival_medium,
    arrival_surge
])


jobs = Job.generate_jobs(
    num_jobs=num_jobs, 
    num_tasks_per_job=num_tasks_per_job,
    time_arrived=arrival_times,
    edge_probability= edge_probability,
    instructions_per_task = [250e5, 500e5],
    data_transfer_range = [512, 1024]
)

# Light
light_gap_test = np.array([round( t , ndigits = 5) for t in np.random.exponential(mean_job_gap*10, jobs_per_phase)])
arrival_light_test = np.cumsum(light_gap_test)
# Medium
medium_gap_test = np.array([round( t , ndigits = 5) for t in np.random.exponential(mean_job_gap*4, jobs_per_phase)])
arrival_medium_test = np.cumsum(medium_gap_test) + arrival_light_test[-1] 
# Surge
surge_gap_test = np.array([round( t , ndigits = 5) for t in np.random.exponential(mean_job_gap, jobs_per_phase)])
arrival_surge_test = np.cumsum(surge_gap_test) + arrival_medium_test[-1] 

arrival_times_test = np.concatenate([
    arrival_light_test,
    arrival_medium_test,
    arrival_surge_test
])


jobs_test = Job.generate_jobs(
    num_jobs=num_jobs, 
    num_tasks_per_job=num_tasks_per_job,
    time_arrived=arrival_times_test,
    edge_probability= edge_probability,
    instructions_per_task = [250e5, 500e5],
    data_transfer_range = [512, 1024]
)

plot_job_dags(jobs_list = jobs)


print("===================================================")
print("BEFORE RUN CHECK")
print(f"ALL TASKS {len([tsk for job in jobs for tsk in job.tasks.values()])}")
print(f"TASK STATES {set([tsk.status for job in jobs for tsk in job.tasks.values()])} : EXPECTED {1, 3}")
print("===================================================")

server_farm = Server_Farm().build_random_server_farms(
    cpu_range = [1024, 2048],
    ram_range = [4096, 4096*4],
    storage_range = [4096, 20000],
    compute_power_range= [1e9, 2e9],
    max_vms_count = 5,
    alphas = [100, 500],
    betas = [2, 5],
    server_count = 4,
    virtual_allocation= [0.9, 0.95],
    bandwidth=[4096, 16384],
    mode = 'LEAST_LOADED'
)

plot_server_network(farm= server_farm)


exp = Experiment(
                infrastructure = server_farm,
                jobs  = jobs,
                scenarios_edges = [min(arrival_medium), min(arrival_surge)],
                schedulers  = [
                                RoundRobinScheduler(),
                                #LeastLoadedScheduler(mode='QUEUE'),
                                LeastLoadedScheduler(mode='CPU'),
                                DataLocalityAwareScheduler(mode='HYBRID'),
                                DataLocalityAwareScheduler(mode='NAIVE'),
                                EnergyAwareScheduler(),
                                RandomAgent(seed= global_seed),
                                DQNAgent()
                            ],
                time_step = 0.005,
                network_overhead_enabled = True,
                power_model = 'DEFAULT',
                evaluation = jobs_test
                
                )

exp.build_environment()
t = time.time()
exp.run_experiment()
print("="*50)
print(f"EXPERIMENT RAN FOR : {round(time.time() - t, ndigits=2)} s")
exp.plot_results()