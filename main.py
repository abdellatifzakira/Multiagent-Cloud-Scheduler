from classes.JobClass import Job
from classes.TaskClass import Task
from classes.VmClass import Vm
from classes.ServerClass import Server
from classes.ServerFarmClass import Server_Farm
from utilities.DAG_handlers import *
from baselines.RoundRobin import RoundRobinScheduler
import igraph as ig
import random
import time
import numpy as np

job_1 =  Job()
    
job_1 = job_1.spawn_job(
        num_tasks= 4,
        cpu_req= [0.01, 0.03, 0.05, 0.03],
        ram_req= [0.01, 0.01, 0.09, 0.03],
        runtime= [10, 10, 25, 10],
        data_transfer_weights= {
         (0,1) : 1,
         (0,2) : 2,
         (1,3) : 3
        }
    )


print(
    get_dag_levels(job_1.dag)
)

print(
    get_dag_level(job_1.dag, 1)
)


