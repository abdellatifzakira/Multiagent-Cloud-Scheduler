from classes.JobClass import Job
from classes.ServerClass import Server
from classes.ServerFarmClass import Server_Farm
from utilities.DAG_handlers import *
from baselines.RoundRobin import RoundRobinScheduler
from comparison.LeastLoaded import LeastLoadedScheduler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from utilities.helpers import *
from utilities.JobManager import JobManager

# Helper
def smooth_by_bins(x, y):
    bin_size= len(x)//50
    x_smooth = []
    y_smooth = []

    for i in range(0, len(y), bin_size):
        y_bin = y[i:i+bin_size]
        x_bin = x[i:i+bin_size]

        if len(y_bin) > 0:
            y_smooth.append(np.mean(y_bin))
            x_smooth.append(np.mean(x_bin))

    return [np.array(x_smooth), np.array(y_smooth)]


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
num_jobs = 360
mean_job_gap = 5
num_tasks_per_job = 5
jobs_per_phase = num_jobs // 3

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
    np.random.exponential(mean_job_gap, jobs_per_phase)
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
    edge_probability= 0.5
)


########################################



seed_LL = 123
np.random.seed(seed_LL)
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

arrival_times_LL = np.concatenate([
    arrival_light,
    arrival_medium,
    arrival_surge
])


jobs_LL = Job.generate_jobs(
    num_jobs=num_jobs, 
    num_tasks_per_job=num_tasks_per_job,
    seed=123,
    time_arrived=arrival_times_LL,
    edge_probability= 0.5
)

plot_job_dags(jobs_list=jobs_RR)


RR = RoundRobinScheduler(
    server_farm=server_farm,
    job_manager = JobManager(jobs = jobs_RR),
)

time_line_RR, cpu_utilization_RR, power_price_RR, server_schedules_RR, data_transfer_RR, sla_violation_RR, workload_std_RR = RR.schedule()

LL = LeastLoadedScheduler(
    server_farm=server_farm,
    job_manager = JobManager(jobs = jobs_LL),
)

time_line_LL, cpu_utilization_LL, power_price_LL, server_schedules_LL, data_transfer_LL, sla_violation_LL, workload_std_LL = LL.schedule()


plt.style.use("seaborn-v0_8-darkgrid")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# ============================================
# CPU UTILIZATION
# ============================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(
        time_line_RR,
        cpu_utilization_RR[server_1],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"Server {server_1.id} : RR"
    )

ax1.plot(
        time_line_LL,
        cpu_utilization_LL[server_1],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"Server {server_1.id} : LL"
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
    smooth_by_bins(time_line_RR, workload_std_RR)[0],
    smooth_by_bins(time_line_RR, workload_std_RR)[1],
    linewidth=2.5,
    marker="^",
    markersize=4,
    color="#00FFFB",
    label = "CPU usage STD RR"
)

ax2.plot(
    smooth_by_bins(time_line_LL, workload_std_LL)[0],
    smooth_by_bins(time_line_LL, workload_std_LL)[1],
    linewidth=2.5,
    marker="^",
    markersize=4,
    color="#00FF11",
    label = "CPU usage STD LL"
)

ax2.axhline(y = np.mean(workload_std_RR),linestyle = "--" , color = "#000787", label = "Avg. CPU usage STD RR")
ax2.axhline(y = np.mean(workload_std_LL), linestyle = "--" , color = "#008A47", label = "Avg. CPU usage STD LL")

ax2.set_title("CPU usage STD Over Time", fontsize=8, fontweight="bold")
ax2.set_ylabel("CPU usage std", fontsize=6)
ax2.set_xlabel("Time", fontsize=6)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(loc="upper left", fontsize=6)
# =============================================
# POWER PRICE
# =============================================
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(
    smooth_by_bins(time_line_RR, power_price_RR)[0],
    smooth_by_bins(time_line_RR, power_price_RR)[1],
    linewidth=2.5,
    marker="s",
    markersize=4,
    color="#FF6B6B",
    label = "Power Price : RR"
)

ax3.plot(
    smooth_by_bins(time_line_LL, power_price_LL)[0],
    smooth_by_bins(time_line_LL, power_price_LL)[1],
    linewidth=2.5,
    marker="s",
    markersize=4,
    color="#77FF6B",
    label = "Power Price : LL"
)

ax3.set_ylim(min(power_price_RR)*0.99, max(power_price_RR)*1.01)
ax3.set_title("Power Price Over Time", fontsize=8, fontweight="bold")
ax3.set_ylabel("Power Price", fontsize=6)
ax3.set_xlabel("Time", fontsize=6)
ax3.grid(True, linestyle="--", alpha=0.4)
ax3.legend(loc="upper left", fontsize=6)

# =============================================
# CUMULATIVE DATA TRANSFER
# =============================================
ax4 = fig.add_subplot(gs[1, 0]) 
ax4.plot(
    time_line_RR,
    np.cumsum(data_transfer_RR),
    linewidth=2.5,
    marker="D",
    markersize=3,
    color="#9370DB",
    label = "Data Transfer Cost : RR"
)

ax4.plot(
    time_line_LL,
    np.cumsum(data_transfer_LL),
    linewidth=2.5,
    marker="D",
    markersize=3,
    color="#82DB70",
    label = "Data Transfer Cost : LL"
)

ax4.set_title("Cumulative Data Transfer", fontsize=8, fontweight="bold")
ax4.set_ylabel("Total Transfer (cumulative)", fontsize=6)
ax4.set_xlabel("Time", fontsize=6)



ax4.grid(True, linestyle="--", alpha=0.4)
ax4.legend(loc="upper left", fontsize=6)

#-------------------------------------
# SLA Violation levels, waiting time
#-------------------------------------

ax5 = fig.add_subplot(gs[1,1])
ax5.plot(
    time_line_RR,
    sla_violation_RR,
    color = "#0350AD",
    label = "SLA violation : RR"
)

ax5.plot(
    time_line_LL,
    sla_violation_LL,
    color = "#FF0000",
    label = "SLA violation : LL"
)

ax5.set_ylabel(
    "SLA Violation (%)",
    fontsize=6
)

ax5.set_xlabel(
    "Time",
    fontsize=6
)

ax5.set_ylim(
    0,
    max(100, max(sla_violation_RR) * 1.2)
)
ax5.legend(loc="upper left", fontsize=6)

# =============================================
# Hosted Tasks per time for each server
# =============================================
ax6 = fig.add_subplot(gs[1, 2])

ax6.plot(
        smooth_by_bins(time_line_RR, server_schedules_RR[server_1])[0],
        smooth_by_bins(time_line_RR, server_schedules_RR[server_1])[1],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"Server {server_1.id} : RR"
    )

ax6.plot(
        smooth_by_bins(time_line_LL, server_schedules_LL[server_1])[0],
        smooth_by_bins(time_line_LL, server_schedules_LL[server_1])[1],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"Server {server_1.id} : LL"
    )

ax6.set_title("Hosted tasks Over Time", fontsize=8, fontweight="bold")
ax6.set_ylabel("Hosted tasks", fontsize=6)
ax6.set_xlabel("Time", fontsize=6)
ax6.grid(True, linestyle="--", alpha=0.4)
ax6.legend(loc="upper left", fontsize=6)





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



plt.legend()
plt.tight_layout()
plt.show()



# =============================================
# SCHEDULING HEATMAP
# =============================================
exit(code=0)
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

