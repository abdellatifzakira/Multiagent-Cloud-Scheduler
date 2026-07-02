from classes.job import Job
from classes.task import Task
from classes.vm import Vm
from classes.server import Server
from classes.server_farm import Server_Farm
from utilities.ready_tasks_queue_generator import queue_generator
import igraph as ig
import random
import time

def print_dag(job):
    print(f"\n  DAG for Job {job.id}:")
    for task_id in job.tasks:
        children = [int(c) for c in job.dag.neighbors(task_id, mode="out")]
        if children:
            print(f"    Task {task_id} → {children}")
        else:
            print(f"    Task {task_id} → (leaf)")

job_count = 2
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
            cpu=round(random.uniform(0.1, 0.5), 3),
            ram=round(random.uniform(0.1, 0.5), 3),
            status=3,      # initialized, not yet ready
            runtime=round(random.uniform(1.0, 10.0), 2)
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
    
    
"""
#Hard coded job for verification of DAG structure and ready tasks :
dag = ig.Graph(directed=True)
dag.add_vertices(5)
dag.add_edges([(0, 1), (2, 1), (1, 3), (2, 3), (3, 4)])

tasks = [
        Task(id=0, job_id=1, cpu=2.0, ram=4.0, status=3, runtime=5.0),  # initialized
        Task(id=1, job_id=1, cpu=1.0, ram=2.0, status=3, runtime=3.0),  # initialized
        Task(id=2, job_id=1, cpu=1.5, ram=3.0, status=3, runtime=4.0),  # initialized
        Task(id=3, job_id=1, cpu=2.5, ram=5.0, status=3, runtime=6.0),  # initialized
        Task(id=4, job_id=1, cpu=3.0, ram=6.0, status=3, runtime=7.0),  # initialized
    ]

job = Job(id=5, dag=dag, tasks=tasks, num_tasks=len(tasks), time_arrived=10.0)
jobs.append(job)
"""
# print results
for job in jobs:
    print(f"\nJob {job.id} | arrived: {job.time_arrived:.2f} | deadline (CPL): {job.deadline:.2f}")
    print_dag(job)
    ready = job.get_first_ready_tasks()
    print(f"  First ready tasks: {[t.id for t in ready]}")

ready_queue = queue_generator(jobs, priority="fifo")
print("\nReady tasks queue (FIFO):")
for task in ready_queue:
    print(f"  Task {task.id} from Job {task.job_id} | runtime: {task.runtime:.2f}")
ready_queue = queue_generator(jobs, priority="runtime")
print("\nReady tasks queue (runtime):")
for task in ready_queue:
    print(f"  Task {task.id} from Job {task.job_id} | runtime: {task.runtime:.2f}")

