import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def smooth_by_bins(x, y, bin_count=50):
    bin_size = max(1, len(x)//bin_count)

    x_smooth = []
    y_smooth = []

    for i in range(0, len(y), bin_size):
        x_bin = x[i:i+bin_size]
        y_bin = y[i:i+bin_size]

        x_smooth.append(np.mean(x_bin))
        y_smooth.append(np.mean(y_bin))

    return np.array(x_smooth), np.array(y_smooth)



def plot_results(results_list, num_bins = 50, scenarios_edges = []):

    plt.style.use("seaborn-v0_8-darkgrid")

    fig = plt.figure(figsize=(18,12))
    gs = GridSpec(2,4, figure=fig, hspace=0.35, wspace=0.3)


    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[0,3])
    ax5 = fig.add_subplot(gs[1,0])
    ax6 = fig.add_subplot(gs[1,1])
    ax7 = fig.add_subplot(gs[1,2])
    ax8 = fig.add_subplot(gs[1,3])

    for res in results_list:

        name = res['NAME']
        t = res['TIMELINE']


        # ==========================
        # CPU STD
        # ==========================

        ax1.plot(
            t,
            res['CPU_STD'],
            linewidth=2,
            label=f"{name}"
        )

        ax1.set_title("CPU Utilization STD")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("STD")
        ax1.legend()



        # ==========================
        # POWER
        # ==========================

        x,y = smooth_by_bins(
            t,
            res['POWER_PRICE'],
            bin_count=num_bins
        )

        ax2.plot(
            x,
            y,
            linewidth=2,
            label=name
        )

        ax2.set_title(f"Power Price\nsmoothed over {num_bins} bins")
        ax2.legend()



        # ==========================
        # DATA TRANSFER
        # ==========================
        
        ax3.plot(
            t,
            res['DATA_TRANSFER'],
            linewidth=2,
            label=name
        )

        ax3.set_title("Data Transfer")
        ax3.legend()

        ax4.plot(
            t,
            res['CUM_DATA_TRANSFER'],
            linewidth=2,
            label=name
        )

        ax4.set_title("Cumulative Data Transfer")
        ax4.legend()



        # ==========================
        # SLA
        # ==========================

        ax5.plot(
            t,
            res['SLA'],
            linewidth=2,
            label=name
        )

        ax5.set_title("SLA Violation (%)")
        ax5.legend()



        # ==========================
        # SLA VAR
        # ==========================

        x,y = smooth_by_bins(
            t,
            res['SLA_VAR'],
            bin_count=num_bins
        )

        ax6.plot(
            x,
            y,
            linewidth=2,
            linestyle="--",
            label=name
        )

        ax6.set_title(f"SLA Variation\nsmoothed over {num_bins} bins")
        ax6.legend()



    # ==========================
    # CPU SERVER LOAD (AGGREGATED)
    # ==========================
    res = [r for r in results_list if r['NAME'] == "RR"][0]
    cpu_values = np.array(list(res['CPU'].values()))
    cpu_mean = np.mean(cpu_values, axis=0)
    cpu_min = np.min(cpu_values, axis=0)
    cpu_max = np.max(cpu_values, axis=0)
    name = res['NAME']
    t = res['TIMELINE']
    ax7.plot(
            t,
            cpu_mean,
            linewidth=2.5,
            color = "#004D00",
            label=f"{name} - Avg CPU"
        )

    ax7.fill_between(
            t,
            cpu_min,
            cpu_max,
            alpha=0.35,
            color = "#00FF00",
            label=f"{name} - CPU range"
        )
    
    ax7.set_ylim(bottom=0, top=1)
    
    ax7.set_title(f"Server CPU Utilization : {name}")
    ax7.legend(fontsize=6)
    
    
    res = [r for r in results_list if r['NAME'] == "LL"][0]
    name = res['NAME']
    t = res['TIMELINE']
    cpu_values = np.array(list(res['CPU'].values()))
    cpu_mean = np.mean(cpu_values, axis=0)
    cpu_min = np.min(cpu_values, axis=0)
    cpu_max = np.max(cpu_values, axis=0)

    ax8.plot(
            t,
            cpu_mean,
            linewidth=2.5,
            color = "#580000",
            label=f"{name} - Avg CPU"
        )

    ax8.fill_between(
            t,
            cpu_min,
            cpu_max,
            alpha=0.35,
            color = "#FF0000",
            label=f"{name} - CPU range"
        )

    ax8.set_ylim(bottom=0, top=1)

    ax8.set_title(f"Server CPU Utilization : {name}")
    ax8.legend(fontsize=6)
    



    for ax in [
        ax1,ax2,ax3,
        ax4,ax5,ax6, ax7, ax8
    ]:
        ax.grid(True, linestyle="--", alpha=0.4)
    
    
    for edge in scenarios_edges :
        
        ax1.axvline(x = edge, ymin=0, color = "#0004ff", linestyle = "--")
        ax2.axvline(x = edge, ymin=0, color = "#0004ff", linestyle = "--")
        ax3.axvline(x = edge, ymin=0, color = "#0004ff", linestyle = "--")
        ax4.axvline(x = edge, ymin=0, color = "#0004ff", linestyle = "--")
        ax5.axvline(x = edge, ymin=0, color = "#0004ff", linestyle = "--")
        ax6.axvline(x = edge, ymin=0, color = "#0004ff", linestyle = "--")
        ax7.axvline(x = edge, ymin=0, color = "#0004ff", linestyle = "--")
        ax8.axvline(x = edge, ymin=0, color = "#0004ff", linestyle = "--")
        


    plt.tight_layout()
    plt.show()