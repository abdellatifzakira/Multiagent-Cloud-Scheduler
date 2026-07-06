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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import networkx as nx
from utilities.helpers import *

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
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3]
)
server_2.spawn_vm_group(
    cpu = [0.55, 0.3],
    ram = [0.3, 0.2]
)
server_3.spawn_vm_group(
    cpu = [0.4],
    ram = [0.5]
)
servers_list = [server_1, server_2, server_3]
#servers_list = [server_1]

server_farm = Server_Farm(
    id =  0,
    servers= servers_list,
    num_servers=len(servers_list)
)


#WORKLOAD


"""

job_1 = Job.spawn_job(
        num_tasks= 4,
        cpu_req= [0.02, 0.02, 0.05, 0.03],
        ram_req= [0.02, 0.05, 0.09, 0.03],
        runtime= [10, 10, 10, 10],
    )

job_2 = Job.spawn_job(
        num_tasks= 4,
        cpu_req= [0.02, 0.02, 0.05, 0.03],
        ram_req= [0.02, 0.05, 0.09, 0.03],
        runtime= [10, 10, 10, 10],
    )

job_3= Job.spawn_job(
        num_tasks= 4,
        cpu_req= [0.02, 0.02, 0.05, 0.03],
        ram_req= [0.02, 0.05, 0.09, 0.03],
        runtime= [10, 10, 10, 10],
    )


jobs = [job_1, job_2]


"""
jobs = Job.generate_jobs(
    num_jobs=3, 
    num_tasks_per_job=4,
    seed=42
    
)

plot_job_dags(jobs=jobs)


RR = RoundRobinScheduler(
    server_farm=server_farm,
    jobs=jobs
)

time_line, task_state, cpu_utilization, power_price, server_schedules, data_transfer = RR.schedule()

plt.style.use("seaborn-v0_8-darkgrid")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# ─────────────────────────────────────────────
# TOP-LEFT: CPU UTILIZATION
# ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for s in cpu_utilization.keys():
    ax1.plot(
        time_line,
        cpu_utilization[s],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"Server {s.id}"
    )
ax1.set_title("CPU Utilization Over Time", fontsize=8, fontweight="bold")
ax1.set_ylabel("CPU Utilization", fontsize=8)
ax1.set_xlabel("Time", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(loc="upper right", fontsize=9)

# ─────────────────────────────────────────────
# TOP-MIDDLE: DATA TRANSFER
# ─────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(
    time_line,
    data_transfer,
    linewidth=2.5,
    marker="^",
    markersize=4,
    color="#FF8C00"
)
ax2.fill_between(time_line, data_transfer, alpha=0.3, color="#FF8C00")
ax2.set_title("Data Transfer Cost Over Time", fontsize=8, fontweight="bold")
ax2.set_ylabel("Data Transfer", fontsize=8)
ax2.set_xlabel("Time", fontsize=8)
ax2.grid(True, linestyle="--", alpha=0.4)

# ─────────────────────────────────────────────
# TOP-RIGHT: POWER PRICE
# ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(
    time_line,
    power_price,
    linewidth=2.5,
    marker="s",
    markersize=4,
    color="#FF6B6B"
)
ax3.set_ylim(min(power_price)*0.99, max(power_price)*1.01)
ax3.fill_between(time_line, power_price, alpha=0.3, color="#FF6B6B")
ax3.set_title("Power Price Over Time", fontsize=8, fontweight="bold")
ax3.set_ylabel("Power Price", fontsize=8)
ax3.set_xlabel("Time", fontsize=8)
ax3.grid(True, linestyle="--", alpha=0.4)

# ─────────────────────────────────────────────
# BOTTOM-LEFT: TASK STATES
# ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for t in task_state.keys():
    ax4.plot(
        time_line,
        task_state[t],
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label = f"task : {t.id}, Job : {t.job_id}"
    )
ax4.set_title("Task States Over Time", fontsize=8, fontweight="bold")
ax4.set_ylabel("State (0=done, 1=ready, 2=running, 3=init)", fontsize=8)
ax4.set_xlabel("Time", fontsize=8)
ax4.grid(True, linestyle="--", alpha=0.4)
ax4.legend(loc="upper right", fontsize=8)

# ─────────────────────────────────────────────
# BOTTOM-MIDDLE: CUMULATIVE DATA TRANSFER
# ─────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
cumulative_transfer = np.cumsum(data_transfer)
ax5.plot(
    time_line,
    cumulative_transfer,
    linewidth=2.5,
    marker="D",
    markersize=3,
    color="#9370DB"
)
ax5.fill_between(time_line, cumulative_transfer, alpha=0.3, color="#9370DB")
ax5.set_title("Cumulative Data Transfer", fontsize=8, fontweight="bold")
ax5.set_ylabel("Total Transfer (cumulative)", fontsize=8)
ax5.set_xlabel("Time", fontsize=8)
ax5.grid(True, linestyle="--", alpha=0.4)

# ─────────────────────────────────────────────
# BOTTOM-RIGHT: SCHEDULING HEATMAP
# ─────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])

time_bins = int(max(max(times) for times in server_schedules.values())) + 1
servers = sorted(server_schedules.keys())
matrix = np.zeros((len(servers), time_bins))

for server_idx, server_id in enumerate(servers):
    for timestamp in server_schedules[server_id]:
        time_bin = int(timestamp)
        matrix[server_idx, time_bin] += 1

im = ax6.imshow(
    matrix,
    cmap='YlOrRd',
    aspect='auto',
    interpolation='nearest'
)
ax6.set_xlabel('Time Steps', fontsize=8)
ax6.set_ylabel('Server ID', fontsize=8)
ax6.set_yticks(range(len(servers)))
ax6.set_yticklabels([f'Server {s}' for s in servers])
ax6.set_title('Scheduling Pattern\n(Round-Robin Distribution)', fontsize=8, fontweight="bold")

cbar = plt.colorbar(im, ax=ax6)
cbar.set_label('# Events', fontsize=8)

fig.suptitle('Cloud Scheduler Simulation - Complete Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.legend()
plt.tight_layout()
plt.show()

#print(f"TOTAL DATA TRANSFER COST = ", np.sum(data_transfer))

#for task in [tsk for job in jobs for tsk in job.tasks.values()] :
#   print(f"TASK ID : {task.id} RUN ON SERVER : {task.server_id}")