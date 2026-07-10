from classes.JobClass import Job
from classes.ServerClass import Server
from classes.ServerFarmClass import Server_Farm
from utilities.DAG_handlers import *
from baselines.RoundRobin import RoundRobinScheduler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from utilities.helpers import *
from utilities.JobManager import JobManager

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
server_4 = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 0.02,
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
    cpu = [0.4,0.5],
    ram = [0.5,0.4]
)

server_4.spawn_vm_group(
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3]
)

servers_list = [server_1, server_2, server_3, server_4]
#servers_list = [server_1]

server_farm = Server_Farm(
    id =  0,
    servers= servers_list,
    num_servers=len(servers_list)
)


#WORKLOAD
num_jobs = 50
mean_job_gap = 10
num_tasks_per_job = 4
arrival_times= np.int32(np.random.exponential(scale=mean_job_gap, size= int(num_jobs)))
arrival_times = np.cumsum(arrival_times)


jobs = Job.generate_jobs(
    num_jobs=num_jobs, 
    num_tasks_per_job=num_tasks_per_job,
    seed=42,
    time_arrived=arrival_times,
)

plot_job_dags(jobs_list=jobs)


RR = RoundRobinScheduler(
    server_farm=server_farm,
    job_manager = JobManager(jobs = jobs),
)

time_line, cpu_utilization, power_price, server_schedules, data_transfer = RR.schedule()









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
# BOTTOM-MIDDLE: CUMULATIVE DATA TRANSFER
# ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
cumulative_transfer = np.cumsum(data_transfer)
ax4.plot(
    time_line,
    cumulative_transfer,
    linewidth=2.5,
    marker="D",
    markersize=3,
    color="#9370DB"
)
ax4.fill_between(time_line, cumulative_transfer, alpha=0.3, color="#9370DB")
ax4.set_title("Cumulative Data Transfer", fontsize=8, fontweight="bold")
ax4.set_ylabel("Total Transfer (cumulative)", fontsize=8)
ax4.set_xlabel("Time", fontsize=8)
ax4.grid(True, linestyle="--", alpha=0.4)


#-------------------------------------
# SLA Violation levels, waiting time
#-------------------------------------

for job in jobs :
    job.end_time = max([tsk.end_time for tsk in job.tasks.values()])

sla_violation_rate = (
    np.sum([
        (job.end_time - job.time_arrived) > job.sla_limit
        for job in jobs
    ])
    / len(jobs)
) * 100


waiting_time = np.mean([
    tsk.start_time - tsk.arrival_time
    for job in jobs
    for tsk in job.tasks.values()
])


ax5 = fig.add_subplot(gs[1,1])
# Left axis: SLA violation rate
ax5.bar(
    [f"SLA Violation : {sla_violation_rate:.2f} %"],
    [sla_violation_rate],
    width=0.4,
    label="Violation Rate (%)"
)

ax5.set_ylabel(
    "SLA Violation (%)",
    fontsize=8
)

ax5.set_ylim(
    0,
    max(100, sla_violation_rate * 1.2)
)

# Right axis: waiting time
ax5_right = ax5.twinx()

ax5_right.bar(
    [f"Average Waiting : {waiting_time:.2f}"],
    [waiting_time],
    width=0.4,
    alpha=0.6,
)

ax5_right.set_ylabel(
    "Waiting Time",
    fontsize=8
)
plt.legend()
plt.tight_layout()
plt.show()





# ─────────────────────────────────────────────
# SCHEDULING HEATMAP
# ─────────────────────────────────────────────

plt.style.use("seaborn-v0_8-darkgrid")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(1, 4, figure=fig, hspace=0.35, wspace=0.3)


ax6 = fig.add_subplot(gs[0,:])


# Flatten all tasks
all_tasks = [
    task
    for job in jobs
    for task in job.tasks.values()
]

num_servers = len(servers_list)
num_tasks = len(all_tasks)

# server x task matrix
schedule_distribution = np.zeros(
    (num_servers, num_tasks)
)

# fill matrix
for task_idx, task in enumerate(all_tasks):
    server_id = task.server_id

    if server_id is not None:
        schedule_distribution[server_id, task_idx] = 1

im = ax6.imshow(
    schedule_distribution,
    cmap='YlOrRd',
    aspect='auto',
    interpolation='nearest'
)

ax6.set_xlabel('Task ID', fontsize=8)
ax6.set_ylabel('Server ID', fontsize=8)
ax6.set_yticks(range(len(servers_list)))
ax6.set_yticklabels([f'Server {s.id}' for s in servers_list])
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