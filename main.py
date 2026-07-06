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
import matplotlib.pylab as plt 



#INFRASTRUCTURE

server_1 = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 0.02,
)
server_2 = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 0.03,
)
server_3 = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 0.05,
)

server_1.spawn_vm_group(
    cpu = [0.2],
    ram = [0.3]
)
server_2.spawn_vm_group(
    cpu = [0.25,0.25,0.25],
    ram = [0.3,0.3,0.3]
)
server_3.spawn_vm_group(
    cpu = [0.1,0.75,0.1],
    ram = [0.1,0.6,0.2]
)

server_farm = Server_Farm(
    id =  0,
    servers=[server_1, server_2, server_3],
    num_servers=3
)


#WORKLOAD

job_1 =  Job()
job_2 = Job()
    
job_1 = job_1.spawn_job(
        num_tasks= 4,
        cpu_req= [0.02, 0.02, 0.05, 0.03],
        ram_req= [0.02, 0.05, 0.09, 0.03],
        runtime= [10, 20, 30, 10],
        data_transfer_weights= {
         (0,2) : 1,
         (1,2) : 2,
         (2,3) : 3
        }
    )


job_2 = job_2.spawn_job(
        num_tasks= 4,
        cpu_req= [0.02, 0.02, 0.05, 0.03],
        ram_req= [0.02, 0.05, 0.09, 0.03],
        runtime= [10, 20, 30, 10],

    )

RR = RoundRobinScheduler(
    server_farm=server_farm,
    jobs=[job_1, job_2]
)

time_line, task_state, cpu_utilization = RR.schedule()

plt.style.use("seaborn-v0_8-darkgrid")

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

# -------------------------
# Top: CPU Utilization
# -------------------------
for s in cpu_utilization.keys():
    ax1.plot(
        time_line,
        cpu_utilization[s],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"Server {s.id}"
    )

ax1.set_title("CPU Utilization Over Time", fontsize=14, fontweight="bold")
ax1.set_ylabel("CPU Utilization", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)


# -------------------------
# Bottom: Task States
# -------------------------
for t in task_state.keys():
    ax2.plot(
        time_line,
        task_state[t],
        linestyle="--",
        linewidth=1.5,
        alpha=0.9,
        label=f"Task {t.id}"
    )

ax2.set_title("Task States Over Time", fontsize=14, fontweight="bold")
ax2.set_xlabel("Time", fontsize=12)
ax2.set_ylabel("State", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()
plt.show()
