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
    storage= 1024
)
server_2 = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 0.03,
    storage= 2048
)
server_3 = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 0.05,
    storage= 4096
)
server_4 = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 0.02,
    storage= 1024
)

server_1.spawn_vm_group(
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3],
    storage=[512, 128, 128]
)
server_2.spawn_vm_group(
    cpu = [0.55, 0.3],
    ram = [0.3, 0.2],
    storage=[512, 512]
)
server_3.spawn_vm_group(
    cpu = [0.4,0.5],
    ram = [0.5,0.4],
    storage=[512, 1024]
)

server_4.spawn_vm_group(
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3],
    storage=[256, 512, 128]
)

servers_list = [server_1, server_2, server_3, server_4]
#servers_list = [server_1, server_2]

server_farm = Server_Farm(
    id =  0,
    servers= servers_list,
    num_servers=len(servers_list)
)


#WORKLOAD
num_jobs = 30
mean_job_gap = 25
num_tasks_per_job = 5

seed = 42
np.random.seed(seed)

jobs_per_phase = num_jobs // 3

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
    np.random.exponential(mean_job_gap, jobs_per_phase)
).astype(int)
arrival_surge = np.cumsum(surge_gap) + arrival_medium[-1]

arrival_times = np.concatenate([
    arrival_light,
    arrival_medium,
    arrival_surge
])


jobs = Job.generate_jobs(
    num_jobs=num_jobs, 
    num_tasks_per_job=num_tasks_per_job,
    seed=42,
    time_arrived=arrival_times,
    edge_probability= 0.5
)

plot_job_dags(jobs_list=jobs)


RR = RoundRobinScheduler(
    server_farm=server_farm,
    job_manager = JobManager(jobs = jobs),
)

time_line, cpu_utilization, power_price, server_schedules, data_transfer, sla_violation, workload_std = RR.schedule()



plt.style.use("seaborn-v0_8-darkgrid")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

# ============================================
# CPU UTILIZATION
# ============================================
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
ax1.set_ylabel("CPU Utilization", fontsize=6)
ax1.set_xlabel("Time", fontsize=6)
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(loc="upper left", fontsize=6)

# =============================================
# CPU UASAGE STD
# =============================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(
    time_line,
    workload_std,
    linewidth=2.5,
    marker="^",
    markersize=4,
    color="#00FFFB"
)
ax2.set_title("CPU usage STD Over Time", fontsize=8, fontweight="bold")
ax2.set_ylabel("CPU usage std", fontsize=6)
ax2.set_xlabel("Time", fontsize=6)
ax2.grid(True, linestyle="--", alpha=0.4)

# =============================================
# DATA TRANSFER
# =============================================
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(
    time_line,
    data_transfer,
    linewidth=2.5,
    marker="^",
    markersize=4,
    color="#FF8C00"
)
ax3.fill_between(time_line, data_transfer, alpha=0.3, color="#FF8C00")
ax3.set_title("Data Transfer Cost Over Time", fontsize=8, fontweight="bold")
ax3.set_ylabel("Data Transfer", fontsize=6)
ax3.set_xlabel("Time", fontsize=6)
ax3.grid(True, linestyle="--", alpha=0.4)

# =============================================
# POWER PRICE
# =============================================
ax4 = fig.add_subplot(gs[0, 3])
ax4.plot(
    time_line,
    power_price,
    linewidth=2.5,
    marker="s",
    markersize=4,
    color="#FF6B6B"
)
ax4.set_ylim(min(power_price)*0.99, max(power_price)*1.01)
ax4.fill_between(time_line, power_price, alpha=0.3, color="#FF6B6B")
ax4.set_title("Power Price Over Time", fontsize=8, fontweight="bold")
ax4.set_ylabel("Power Price", fontsize=6)
ax4.set_xlabel("Time", fontsize=6)
ax4.grid(True, linestyle="--", alpha=0.4)


# =============================================
# CUMULATIVE DATA TRANSFER
# =============================================
ax5 = fig.add_subplot(gs[1, 0])
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
ax5.set_ylabel("Total Transfer (cumulative)", fontsize=6)
ax5.set_xlabel("Time", fontsize=6)



ax5.grid(True, linestyle="--", alpha=0.4)


#-------------------------------------
# SLA Violation levels, waiting time
#-------------------------------------

ax6 = fig.add_subplot(gs[1,1])
ax6.plot(
    time_line,
    sla_violation
)
ax6.set_ylabel(
    "SLA Violation (%)",
    fontsize=6
)

ax6.set_xlabel(
    "Time",
    fontsize=6
)

ax6.set_ylim(
    0,
    max(100, max(sla_violation) * 1.2)
)


# =============================================
# Hosted Tasks per time for each server
# =============================================
ax7 = fig.add_subplot(gs[1, 2])
for s in server_schedules.keys():

    ax7.plot(
        time_line,
        server_schedules[s],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"Server {s.id}"
    )
ax7.set_title("Hosted tasks Over Time", fontsize=8, fontweight="bold")
ax7.set_ylabel("Hosted tasks", fontsize=6)
ax7.set_xlabel("Time", fontsize=6)
ax7.grid(True, linestyle="--", alpha=0.4)
ax7.legend(loc="upper left", fontsize=6)





ax1.axvline(x=min(arrival_medium), ymin=0, color = "#0004ff", linestyle = "--")
ax1.axvline(x=min(arrival_surge), ymin=0, color = "#0004ff", linestyle = "--")
ax2.axvline(x=min(arrival_medium), ymin=0, color = "#0004ff", linestyle = "--")
ax2.axvline(x=min(arrival_surge), ymin=0, color = "#0004ff", linestyle = "--")
ax3.axvline(x=min(arrival_medium), ymin=0, color = "#0004ff", linestyle = "--")
ax3.axvline(x=min(arrival_surge), ymin=0, color = "#0004ff", linestyle = "--")
ax4.axvline(x=min(arrival_medium), ymin=0, color = "#0004ff", linestyle = "--")
ax4.axvline(x=min(arrival_surge), ymin=0, color = "#0004ff", linestyle = "--")
ax5.axvline(x=min(arrival_medium), ymin=0, color = "#0004ff", linestyle = "--")
ax5.axvline(x=min(arrival_surge), ymin=0, color = "#0004ff", linestyle = "--")
ax6.axvline(x=min(arrival_medium), ymin=0, color = "#0004ff", linestyle = "--")
ax6.axvline(x=min(arrival_surge), ymin=0, color = "#0004ff", linestyle = "--")
ax7.axvline(x=min(arrival_medium), ymin=0, color = "#0004ff", linestyle = "--")
ax7.axvline(x=min(arrival_surge), ymin=0, color = "#0004ff", linestyle = "--")



plt.legend()
plt.tight_layout()
plt.show()



# =============================================
# SCHEDULING HEATMAP
# =============================================

if len(jobs) <= 4 :

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(1, 4, figure=fig, hspace=0.35, wspace=0.3)


    ax8 = fig.add_subplot(gs[0,:])


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
        server_id = task.server.id

        if server_id is not None:
            schedule_distribution[server_id, task_idx] = 1

    im = ax8.imshow(
        schedule_distribution,
        cmap='YlOrRd',
        aspect='auto',
        interpolation='nearest'
    )

    ax8.set_xlabel('Task ID', fontsize=8)
    ax8.set_ylabel('Server ID', fontsize=8)
    ax8.set_yticks(range(len(servers_list)))
    ax8.set_yticklabels([f'Server {s.id}' for s in servers_list])
    ax8.set_title('Scheduling Pattern\n(Round-Robin Distribution)', fontsize=8, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax8)
    cbar.set_label('# Events', fontsize=8)

    fig.suptitle('Cloud Scheduler Simulation - Complete Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #print(f"TOTAL DATA TRANSFER COST = ", np.sum(data_transfer))

    #for task in [tsk for job in jobs for tsk in job.tasks.values()] :
    #   print(f"TASK ID : {task.id} RUN ON SERVER : {task.server_id}")

