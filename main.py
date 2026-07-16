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
import random

# Helper
def smooth_by_bins(x, y, bin_count = 50):
    bin_size= len(x)//bin_count
    x_smooth = []
    y_smooth = []

    for i in range(0, len(y), bin_size,):
        y_bin = y[i:i+bin_size]
        x_bin = x[i:i+bin_size]

        if len(y_bin) > 0:
            y_smooth.append(np.mean(y_bin))
            x_smooth.append(np.mean(x_bin))

    return [np.array(x_smooth), np.array(y_smooth)]


#INFRASTRUCTURE RR
server_1_RR = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 25,
    beta=1.2,
    storage= 512
)
server_2_RR = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 50,
    beta=1.5,
    storage= 2048
)
server_3_RR = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 40,
    beta= 2,
    storage= 1024
)
server_4_RR = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 15,
    beta= 1.8,
    storage= 1024
)

server_1_RR.spawn_vm_group(
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3],
    storage=[128, 128, 128]
)
server_2_RR.spawn_vm_group(
    cpu = [0.55, 0.3],
    ram = [0.3, 0.2],
    storage=[512, 512]
)
server_3_RR.spawn_vm_group(
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3],
    storage=[512, 128, 128]
)

server_4_RR.spawn_vm_group(
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3],
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
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 25,
    beta=1.2,
    storage= 512
)
server_2_LL = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 50,
    beta=1.5,
    storage= 2048
)
server_3_LL = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 40,
    beta= 2,
    storage= 1024
)
server_4_LL = Server(
    c_cpu=1.0,
    c_ram=1.0,
    alpha = 15,
    beta= 1.8,
    storage= 1024
)

server_1_LL.spawn_vm_group(
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3],
    storage=[128, 128, 128]
)
server_2_LL.spawn_vm_group(
    cpu = [0.55, 0.3],
    ram = [0.3, 0.2],
    storage=[512, 512]
)
server_3_LL.spawn_vm_group(
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3],
    storage=[512, 128, 128]
)

server_4_LL.spawn_vm_group(
    cpu = [0.2, 0.3, 0.4],
    ram = [0.3, 0.3, 0.3],
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
num_tasks_per_job = 15
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
    edge_probability= 0.05
)


########################################



seed_LL = 42
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
    np.random.exponential(mean_job_gap/2, jobs_per_phase)
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
    seed=42,
    time_arrived=arrival_times_LL,
    edge_probability= 0.05
)

all_jobs = jobs_LL + jobs_RR
random.shuffle(all_jobs)
plot_job_dags(jobs_list = all_jobs)


RR = RoundRobinScheduler(
    server_farm= server_farm_RR,
    job_manager = JobManager(jobs = jobs_RR),
)

mode = 'QUEUE'
sorting='FIFO'
LL = LeastLoadedScheduler(
    server_farm= server_farm_LL,
    job_manager = JobManager(jobs = jobs_LL),
    mode= mode,
    sorting=sorting
)

print("RoundRobin : SCHEDULING - STARTS")

(time_line_RR, cpu_utilization_RR,
 power_price_RR, server_schedules_RR,
 data_transfer_RR, sla_violation_RR,
 sla_violation_var_RR, workload_std_RR) = RR.schedule()



print(f"LeastLoaded {mode = }, {sorting = } : SCHEDULING - STARTS")

(time_line_LL, cpu_utilization_LL,
power_price_LL, server_schedules_LL,
data_transfer_LL, sla_violation_LL,
sla_violation_var_LL , workload_std_LL ) = LL.schedule()


##########################################



def evaluate_scheduler(
        name,
        scheduler,
        jobs,
        time_line,
        cpu_usage,
        power_price,
        data_transfer,
        sla_violation,
        workload_std
):

    print("\n" + "="*60)
    print(f"METRICS : {name}")
    print("="*60)


    # =========================
    # Job completion metrics
    # =========================

    finished_jobs = scheduler.get_finished_jobs()

    print("\n--- JOB COMPLETION ---")

    print(
        f"Finished jobs : {len(finished_jobs)}/{len(jobs)}"
    )

    if finished_jobs:

        completion_times = np.array([
            job.end_time - job.time_arrived
            for job in finished_jobs
        ])

        print(
            f"Makespan : {max(job.end_time for job in finished_jobs)}"
        )

        print(
            f"Average completion time : {np.mean(completion_times):.2f}"
        )

        print(
            f"Median completion time : {np.median(completion_times):.2f}"
        )

        print(
            f"P95 completion time : {np.percentile(completion_times,95):.2f}"
        )


    # =========================
    # SLA metrics
    # =========================

    print("\n--- SLA ---")

    print(
        f"Final SLA violation : {sla_violation[-1]:.2f}%"
    )

    print(
        f"Average SLA violation : {np.mean(sla_violation):.2f}%"
    )

    print(
        f"Maximum SLA violation : {np.max(sla_violation):.2f}%"
    )


    if finished_jobs:

        sla_ratios = np.array([
            (job.end_time-job.time_arrived)
            /
            job.sla_limit
            for job in finished_jobs
        ])

        print(
            f"Average SLA delay ratio : {np.mean(sla_ratios):.3f}"
        )

        print(
            f"P95 SLA delay ratio : {np.percentile(sla_ratios,95):.3f}"
        )


    # =========================
    # CPU metrics
    # =========================

    print("\n--- CPU ---")

    all_cpu = np.concatenate(
        [
            np.array(v)
            for v in cpu_usage.values()
        ]
    )

    print(
        f"Average CPU utilization : {np.mean(all_cpu):.3f}"
    )

    print(
        f"Peak CPU utilization : {np.max(all_cpu):.3f}"
    )

    print(
        f"CPU utilization std : {np.mean(workload_std):.4f}"
    )

    print(
        f"Maximum workload std : {np.max(workload_std):.4f}"
    )


    # =========================
    # Power
    # =========================

    print("\n--- POWER ---")

    print(
        f"Total power cost : {np.sum(power_price):.2f}"
    )

    print(
        f"Average power cost : {np.mean(power_price):.3f}"
    )

    print(
        f"Peak power cost : {np.max(power_price):.3f}"
    )


    # =========================
    # Network
    # =========================

    print("\n--- NETWORK ---")

    print(
        f"Total data transfer : {np.sum(data_transfer):.2f}"
    )

    if len(finished_jobs)>0:
        print(
            f"Data transfer/job : "
            f"{np.sum(data_transfer)/len(finished_jobs):.3f}"
        )


    # =========================
    # Time
    # =========================

    print("\n--- SIMULATION ---")

    print(
        f"Simulation steps : {len(time_line)}"
    )

    print(
        "="*60
    )
    


evaluate_scheduler(
    "Round Robin",
    RR,
    jobs_RR,
    time_line_RR,
    cpu_utilization_RR,
    power_price_RR,
    data_transfer_RR,
    sla_violation_RR,
    workload_std_RR
)


evaluate_scheduler(
    f"Least Loaded {mode}",
    LL,
    jobs_LL,
    time_line_LL,
    cpu_utilization_LL,
    power_price_LL,
    data_transfer_LL,
    sla_violation_LL,
    workload_std_LL
)





plt.style.use("seaborn-v0_8-darkgrid")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# ============================================
# CPU UTILIZATION
# ============================================
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(
        time_line_RR,
        cpu_utilization_RR[server_1_RR],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"Server {server_1_RR.id} : RR"
    )

ax1.plot(
        time_line_LL,
        cpu_utilization_LL[server_1_LL],
        linewidth=2.5,
        marker="o",
        markersize=3,
        label=f"Server {server_1_LL.id} : LL"
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
# SLA Violation levels
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
# SLA : rate of change
# =============================================
num_bins = 500
ax6 = fig.add_subplot(gs[1, 2])

ax6.plot(
    smooth_by_bins(time_line_LL,sla_violation_var_LL, bin_count = num_bins)[0],
    smooth_by_bins(time_line_LL,sla_violation_var_LL, bin_count = num_bins)[1],
    color = "#FF0000",
    linestyle = '--',
    label = "SLA violation variation : LL"
)
ax6.plot(
    smooth_by_bins(time_line_RR,sla_violation_var_RR, bin_count = num_bins)[0],
    smooth_by_bins(time_line_RR,sla_violation_var_RR, bin_count = num_bins)[1],
    color = "#0350AD",
    linestyle = '--',
    label = "SLA violation variation : RR"
)

ax6.set_title(
    f"SLA Violation Variation : smoothed over {num_bins} bins",
    fontsize=8,
    fontweight="bold"
)
ax6.set_ylabel(
    "Change in SLA Violation Rate (%)",
    fontsize=6
)

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

