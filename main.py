from classes.JobClass import Job
from classes.ServerClass import Server
from classes.ServerFarmClass import Server_Farm
from utilities.DAG_handlers import *
from baselines.RoundRobin import RoundRobinScheduler
from comparison.LeastLoaded import LeastLoadedScheduler
import numpy as np
from utilities.helpers import *
from utilities.JobManager import JobManager
from environment.EnvironmentClass import Environment
from environment.MetricsManager import MetricsManager
from utilities.ResultPlotter import plot_results
import random
import copy

server_1_RR = Server(
    c_cpu=256,
    c_ram=256,
    alpha=25,
    beta=1.2,
    storage=512
)

server_2_RR = Server(
    c_cpu=512,
    c_ram=512,
    alpha=50,
    beta=1.5,
    storage=2048
)

server_3_RR = Server(
    c_cpu=384,
    c_ram=384,
    alpha=40,
    beta=2,
    storage=1024
)

server_4_RR = Server(
    c_cpu=256,
    c_ram=256,
    alpha=15,
    beta=1.8,
    storage=1024
)

server_1_RR.spawn_vm_group(
    cpu=[50, 75, 100],   
    ram=[75, 75, 75],
    storage=[128, 128, 128]
)


server_2_RR.spawn_vm_group(
    cpu=[280, 150],     
    ram=[150, 100],
    storage=[512, 512]
)


server_3_RR.spawn_vm_group(
    cpu=[75, 115, 150],    
    ram=[115, 115, 115],
    storage=[512, 128, 128]
)


server_4_RR.spawn_vm_group(
    cpu=[50, 75, 100],
    ram=[75, 75, 75],
    storage=[256, 512, 128]
)
servers_list_RR = [server_1_RR, server_2_RR, server_3_RR, server_4_RR]


server_farm_RR = Server_Farm(
    id =  0,
    servers= servers_list_RR,
    num_servers=len(servers_list_RR)
)


#INFRASTRUCTURE LL

server_1_LL = Server(
    c_cpu=256,
    c_ram=256,
    alpha=25,
    beta=1.2,
    storage=512
)

server_2_LL = Server(
    c_cpu=512,
    c_ram=512,
    alpha=50,
    beta=1.5,
    storage=2048
)

server_3_LL = Server(
    c_cpu=384,
    c_ram=384,
    alpha=40,
    beta=2,
    storage=1024
)

server_4_LL = Server(
    c_cpu=256,
    c_ram=256,
    alpha=15,
    beta=1.8,
    storage=1024
)

server_1_LL.spawn_vm_group(
    cpu=[50, 75, 100],     
    ram=[75, 75, 75],
    storage=[128, 128, 128]
)


server_2_LL.spawn_vm_group(
    cpu=[280, 150],       
    ram=[150, 100],
    storage=[512, 512]
)


server_3_LL.spawn_vm_group(
    cpu=[75, 115, 150],    
    ram=[115, 115, 115],
    storage=[512, 128, 128]
)


server_4_LL.spawn_vm_group(
    cpu=[50, 75, 100],
    ram=[75, 75, 75],
    storage=[256, 512, 128]
)

servers_list_LL = [server_1_LL, server_2_LL, server_3_LL, server_4_LL]


server_farm_LL = Server_Farm(
    id =  0,
    servers= servers_list_LL,
    num_servers=len(servers_list_LL)
)


#WORKLOAD
num_jobs = 180
mean_job_gap = 5
num_tasks_per_job = 8
jobs_per_phase = num_jobs // 3
edge_probability =  0.2

seed_RR = 42
np.random.seed(seed_RR)
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


jobs_RR = Job.generate_jobs(
    num_jobs=num_jobs, 
    num_tasks_per_job=num_tasks_per_job,
    seed=42,
    time_arrived=arrival_times_RR,
    edge_probability= edge_probability
)
jobs_LL = copy.deepcopy(jobs_RR)

plot_job_dags(jobs_list = jobs_RR)

Environment_RR = Environment(server_farm= server_farm_RR,
                             metrics_manager= MetricsManager(server_farm_RR, jobs= jobs_RR),
                             job_manager = JobManager(jobs=jobs_RR),
                             scheduler=RoundRobinScheduler()
                             )

mode = 'CPU'
sorting='FIFO'
Environment_LL = Environment(server_farm= server_farm_LL,
                             metrics_manager= MetricsManager(server_farm_LL, jobs= jobs_LL),
                             job_manager = JobManager(jobs=jobs_LL),
                             scheduler= LeastLoadedScheduler(mode= mode, sorting=sorting)
                             )



print("RoundRobin : SCHEDULING - STARTS")
results_RR = Environment_RR.run()
print(f"LeastLoaded {mode = }, {sorting = } : SCHEDULING - STARTS")
results_LL = Environment_LL.run()
results_RR['NAME'] = "RR"
results_LL['NAME'] = "LL"



print(f"TOTAL SUCCESSFUL JOBS RR : {sum(j.success for j in jobs_RR)}/{num_jobs}")
print(f"TOTAL SUCCESSFUL JOBS LL : {sum(j.success for j in jobs_LL)}/{num_jobs}")

scenarios_edges = [max(arrival_light), max(arrival_medium)]

plot_results([results_RR, results_LL], num_bins = 100, scenarios_edges = scenarios_edges)