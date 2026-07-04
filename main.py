from classes import vm
from classes.job import Job
from classes.task import Task
from classes.VmClass import Vm
from classes.ServerClass import Server
from classes.server_farm import Server_Farm
from utilities.DAG_handlers import queue_creator
from baselines.RoundRobin import RoundRobinScheduler
import igraph as ig
import random
import time
import numpy as np

def print_dag(job):
    print(f"\n  DAG for Job {job.id}:")
    for task_id in job.tasks:
        children = [int(c) for c in job.dag.neighbors(task_id, mode="out")]
        if children:
            print(f"Task {task_id} → {children}")
        else:
            print(f"Task {task_id} → (leaf)")


job_count = 3
num_tasks = 5          # fixed number of tasks per job
jobs = []

for job_id in range(job_count):
    # FIX: DAG vertices must match number of tasks exactly
    dag = ig.Graph(directed=True)
    dag.add_vertices(num_tasks)

    # FIX: only generate edges between vertices that actually exist
    # (i,j) with j>i guarantees forward-only edges = valid DAG always
    all_edges = [(i, j) for i in range(num_tasks)
                         for j in range(i + 1, num_tasks)]
    random.shuffle(all_edges)
    selected_edges = all_edges[:random.randint(1, len(all_edges))]
    dag.add_edges(selected_edges)

    tasks = [
        Task(
            id=i,
            job_id=job_id,
            cpu=round(random.uniform(0.01, 0.25), 3),
            ram=round(random.uniform(0.01, 0.25), 3),
            status=3,      # initialized, not yet ready
            runtime=round(random.uniform(1.0, 20.0), 2)
        )
        for i in range(num_tasks)
    ]

    job = Job(
        id=job_id,
        dag=dag,
        tasks=tasks,
        num_tasks=num_tasks,
        time_arrived=time.time()
    )
    jobs.append(job)
    
# print results
for job in jobs:
    print(f"\nJob {job.id} | arrived: {job.time_arrived:.2f} | deadline (CPL): {job.deadline:.2f}")
    print_dag(job)
    ready = job.get_first_ready_tasks()
    print(f"First ready tasks: {[t.id for t in ready]}")

server_1 =  Server(id = 1, server_farm_id=1, vms = [], c_cpu=1.0, c_ram=1.0, alpha = 0.5, beta = 0.1)
server_1.spawn_random_vms(cpu = [0.25, 0.3, 0.3], ram= [0.3,0.3,0.3])
server_2 =  Server(id = 2, server_farm_id=1, vms = [], c_cpu=1.0, c_ram=1.0, alpha = 0.5, beta = 0.1)
server_2.spawn_random_vms(cpu = [0.5, 0.3], ram= [0.45,0.3])

