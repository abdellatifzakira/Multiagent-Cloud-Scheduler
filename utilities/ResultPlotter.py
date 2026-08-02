import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import math


def find_closest_factors(k):
    """
    Find n,m such that:
    n*m >= k
    and grid is as square as possible
    """

    best = None
    best_diff = float("inf")

    for n in range(1, k+1):
        m = math.ceil(k/n)

        diff = abs(m-n)

        if diff < best_diff:
            best = (n,m)
            best_diff = diff

    return best



def smooth_by_bins(x, y, bin_count=50):

    bin_size = max(1,len(x)//bin_count)

    xs=[]
    ys=[]

    for i in range(0,len(y),bin_size):

        xs.append(np.mean(x[i:i+bin_size]))
        ys.append(np.mean(y[i:i+bin_size]))

    return np.array(xs), np.array(ys)





def plot_metrics(results_list,
                 num_bins=50,
                 scenarios_edges=[]):


    plt.style.use("seaborn-v0_8-darkgrid")


    metrics = [
        ("CPU_STD","Standard Deviation\nVirtual CPU Efficiency", "STD CPU Efficiency"),
        ("POWER_PRICE","Power Price Over Time\n(Smoothed)", "Power Price"),
        ("CUM_DATA_TRANSFER","Cumulative Data Transfer Over Time", "Data Transfer"),
        ("SLA","SLA Violations", "Number Of Jobs"),
        ("JOB_MEAN_COMPLETION_TIME", "Mean Completion\nTime Per Job", "Completion Time"),
        ("NETWORK_LATENCY", "Cumulative Network Latency", "Network Latency")
    ]


    rows,cols = find_closest_factors(len(metrics))


    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6*cols,4*rows),
        squeeze=False
    )


    axes = axes.flatten()



    for idx,(metric,title, label) in enumerate(metrics):

        ax = axes[idx]


        for res in results_list:

            t=res["TIMELINE"]
            y=res[metric]


            if metric=="POWER_PRICE":

                t,y=smooth_by_bins(
                    t,
                    y,
                    num_bins
                )
            
            if metric=="POWER_PRICE":
                ax.plot(
                    t,
                    y,
                    linewidth=2,
                    label=res["NAME"] + ' : ' + f'{round(np.mean(y), ndigits=2)}'
                )
            else :
                ax.plot(
                        t,
                        y,
                        linewidth=2,
                        label=res["NAME"]
                                )


        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(label)


        ax.legend()



    # remove empty plots

    for i in range(len(metrics),len(axes)):
        axes[i].remove()


    for edge in scenarios_edges:
        for ax in axes:
            ax.axvline(
                edge,
                linestyle="--"
            )


    plt.tight_layout()
    plt.show()


    schedulers=len(results_list)

    rows,cols=find_closest_factors(schedulers)


    fig,axes=plt.subplots(
        rows,
        cols,
        figsize=(6*cols,4*rows),
        squeeze=False
    )


    axes=axes.flatten()



    for idx,res in enumerate(results_list):

        ax=axes[idx]

        cpu_values=np.array(
            list(res["CPU"].values())
        )


        cpu_mean=np.mean(
            cpu_values,
            axis=0
        )


        cpu_min=np.min(
            cpu_values,
            axis=0
        )


        cpu_max=np.max(
            cpu_values,
            axis=0
        )


        t=res["TIMELINE"]


        ax.plot(
            t,
            cpu_mean,
            linewidth=2,
            label="Average CPU Efficiency",
            color = "#008B33"
        )


        ax.fill_between(
            t,
            cpu_min,
            cpu_max,
            alpha=0.25,
            label="CPU Efficiency range",
            color = "#FF0000"
        )


        ax.set_ylim(0,1)


        ax.set_title(
            f"Virtual CPU Efficiency : {res['NAME']}\n"
            f"Mean STD = {np.mean(res['CPU_STD']):.3f}"
        )


        ax.set_xlabel("Time")
        ax.set_ylabel("CPU Efficiency")


        ax.legend()



    for i in range(schedulers,len(axes)):
        axes[i].remove()



    for edge in scenarios_edges:
        for ax in axes:
            ax.axvline(
                edge,
                linestyle="--"
            )


    plt.tight_layout()
    plt.show()