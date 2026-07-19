from classes.JobClass import Job
from classes.ServerFarmClass import Server_Farm
from utilities.DAG_handlers import *
from baselines.RoundRobin import RoundRobinScheduler
from comparison.LeastLoaded import LeastLoadedScheduler
from comparison.LeastCommunication import DataLocalityAwareScheduler
import numpy as np
import random
from utilities.helpers import *
from ExperimentRunner import Experiment

# REPRODUCIBILITY
global_seed = 42
random.seed(global_seed)
np.random.seed(global_seed)

#WORKLOAD
num_jobs = 300
mean_job_gap = 5
num_tasks_per_job = 5
jobs_per_phase = num_jobs // 3
edge_probability =  0.75 # controls how fuzzy the jobs are

# Light
light_gap = np.ceil(
    np.random.exponential(mean_job_gap*8, jobs_per_phase)
).astype(int)
arrival_light = np.cumsum(light_gap)
# Medium
medium_gap = np.ceil(
    np.random.exponential(mean_job_gap*4, jobs_per_phase)
).astype(int)
arrival_medium = np.cumsum(medium_gap) + arrival_light[-1]
# Surge
surge_gap = np.ceil(
    np.random.exponential(mean_job_gap/2, jobs_per_phase)
).astype(int)
arrival_surge = np.cumsum(surge_gap) + arrival_medium[-1]

arrival_times_RR = np.concatenate([
    arrival_light,
    arrival_medium,
    arrival_surge
])


jobs = Job.generate_jobs(
    num_jobs=num_jobs, 
    num_tasks_per_job=num_tasks_per_job,
    time_arrived=arrival_times_RR,
    edge_probability= edge_probability
)
plot_job_dags(jobs_list = jobs)



server_farm = Server_Farm().build_random_server_farms(
    cpu_range = [256, 1024],
    ram_range = [512, 4096],
    storage_range = [4096, 20000],
    max_vms_count = 4,
    alphas = [100, 500],
    betas = [2, 5],
    server_count = 3,
    virtual_allocation= [0.9, 0.95]
)

exp = Experiment(
    infrastructure = server_farm,
    jobs  = jobs,
    scenarios_edges = [min(arrival_medium), min(arrival_surge)],
    schedulers  = [RoundRobinScheduler(), LeastLoadedScheduler(mode='CPU'), DataLocalityAwareScheduler()]
           )

exp.build_environment()
exp.run_experiment()
exp.plot_results()