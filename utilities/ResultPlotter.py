import matplotlib.pyplot as plt
import numpy as np
import math


def find_closest_factors(k):
    """
    Find n, m such that:
        n * m >= k
    while keeping the grid approximately square.
    """
    best = None
    best_diff = float("inf")

    for n in range(1, k + 1):
        m = math.ceil(k / n)
        diff = abs(m - n)

        if diff < best_diff:
            best = (n, m)
            best_diff = diff

    return best


def smooth_by_bins(x, y, bin_count=50):
 
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0:
        return x, y

    bin_size = max(1, len(x) // bin_count)

    xs = []
    ys = []

    for i in range(0, len(y), bin_size):
        xs.append(np.mean(x[i:i + bin_size]))
        ys.append(np.mean(y[i:i + bin_size]))

    return np.asarray(xs), np.asarray(ys)

def configure_plot_style():


    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,

        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "legend.fontsize": 9,

        "axes.titleweight": "bold",

        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "lines.linewidth": 2.0,

        "figure.autolayout": False,
    })

def add_scenario_edges(ax, scenarios_edges):

    if not scenarios_edges:
        return

    for edge in scenarios_edges:
        ax.axvline(
            edge,
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
            zorder=1,
        )


def plot_time_metrics(
    results_list,
    num_bins=50,
    scenarios_edges=None,
):

    if scenarios_edges is None:
        scenarios_edges = []

    metrics = [
        (
            "CPU_STD",
            "CPU Efficiency Variability",
            "CPU STD",
            False,
        ),
        (
            "POWER",
            "Power Consumption",
            "Power",
            True,
        ),
        (
            "CUM_DATA_TRANSFER",
            "Cumulative Data Transfer",
            "Data Transfer",
            False,
        ),
        (
            "NETWORK_LATENCY",
            "Cumulative Network Latency",
            "Network Latency",
            False,
        ),
    ]

    rows, cols = find_closest_factors(len(metrics))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(7.0 * cols, 4.8 * rows),
        squeeze=False,
    )

    axes = axes.flatten()

    for idx, (metric, title, ylabel, smooth) in enumerate(metrics):

        ax = axes[idx]

        for res in results_list:

            if metric not in res:
                continue

            if "TIMELINE" not in res:
                continue

            t = np.asarray(res["TIMELINE"])
            y = np.asarray(res[metric])

            if len(t) == 0 or len(y) == 0:
                continue

            if smooth:
                t, y = smooth_by_bins(
                    t,
                    y,
                    num_bins,
                )

                energy = np.trapezoid(y, t) / 1000

                label = (
                    f"{res['NAME']} "
                    f"({energy:.2f} kJ)"
                )
            else:
                label = res["NAME"]

            ax.plot(
                t,
                y,
                label=label,
                linewidth=2.0,
            )

        add_scenario_edges(
            ax,
            scenarios_edges,
        )

        ax.set_title(title)

        ax.set_xlabel("Simulation Time")
        ax.set_ylabel(ylabel)

        ax.margins(x=0.02)

        ax.legend(
            loc="best",
            frameon=True,
        )

    for i in range(len(metrics), len(axes)):
        axes[i].remove()

    fig.suptitle(
        "System Performance Over Simulation Time",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    plt.show()


def plot_performance_metrics(results_list):

    metrics_available = []

    for res in results_list:

        if "JOB_MEAN_COMPLETION_TIME" in res:
            metrics_available.append(
                "JOB_MEAN_COMPLETION_TIME"
            )

        break

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.2),
    )


    ax = axes[0]

    for res in results_list:

        if "JOB_MEAN_COMPLETION_TIME" not in res:
            continue

        t = np.asarray(res["TIMELINE"])
        y = np.asarray(
            res["JOB_MEAN_COMPLETION_TIME"]
        )

        if len(t) == 0 or len(y) == 0:
            continue

        ax.plot(
            t,
            y,
            linewidth=2.0,
            label=res["NAME"],
        )

    ax.set_title(
        "Mean Job Completion Time"
    )

    ax.set_xlabel(
        "Simulation Time"
    )

    ax.set_ylabel(
        "Mean Completion Time"
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    ax.margins(x=0.02)

 
    ax = axes[1]

    for res in results_list:

        if "REWARD" not in res:
            continue

        reward = np.asarray(
            res["REWARD"]
        )

        if len(reward) == 0:
            continue

        episodes = np.arange(
            len(reward)
        )

        ax.plot(
            episodes,
            reward,
            linewidth=2.0,
            label=res["NAME"],
        )

    ax.set_title(
        "Training Reward"
    )

    ax.set_xlabel(
        "Episode"
    )

    ax.set_ylabel(
        "Reward"
    )

    ax.legend(
        loc="best",
        frameon=True,
    )

    ax.margins(x=0.02)

    fig.suptitle(
        "Scheduling Performance",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    plt.show()



def plot_sla_violations(results_list):

    names = []
    sla_rates = []

    for res in results_list:

        if "FINAL_SLA" not in res:
            continue

        value = res["FINAL_SLA"]

        # Handle scalar values.
        if np.isscalar(value):
            rate = float(value)

        else:
            values = np.asarray(value)

            if values.size == 0:
                continue

            rate = float(
                values.reshape(-1)[-1]
            )

        names.append(
            res["NAME"]
        )

        sla_rates.append(
            rate * 100.0
        )

    if not names:
        return

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(names) * 2.0),
            5.5,
        )
    )

    x = np.arange(
        len(names)
    )

    bars = ax.bar(
        x,
        sla_rates,
        width=0.65,
    )

    ax.set_title(
        "Final SLA Violation Rate",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Scheduler"
    )

    ax.set_ylabel(
        "SLA Violation Rate (%)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        names,
        rotation=0,
        ha="center",
    )

    # Start from zero because this is a rate.
    ax.set_ylim(
        bottom=0
    )

    # Value labels above bars.
    for bar, value in zip(
        bars,
        sla_rates,
    ):

        ax.annotate(
                    f"{value:.1f}%",
                    xy=(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                    ),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

    ax.margins(
        x=0.08
    )

    fig.tight_layout()

    plt.show()

def plot_cpu_efficiency(
    results_list,
    scenarios_edges=None,
):

    if scenarios_edges is None:
        scenarios_edges = []

    schedulers = len(
        results_list
    )

    if schedulers == 0:
        return

    rows, cols = find_closest_factors(
        schedulers
    )

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(
            7.0 * cols,
            4.8 * rows,
        ),
        squeeze=False,
    )

    axes = axes.flatten()

    for idx, res in enumerate(
        results_list
    ):

        ax = axes[idx]

        if "CPU" not in res:
            ax.remove()
            continue

        cpu_values = np.asarray(
            list(
                res["CPU"].values()
            )
        )

        if cpu_values.size == 0:
            ax.remove()
            continue

        cpu_mean = np.mean(
            cpu_values,
            axis=0,
        )

        cpu_min = np.min(
            cpu_values,
            axis=0,
        )

        cpu_max = np.max(
            cpu_values,
            axis=0,
        )

        t = np.asarray(
            res["TIMELINE"]
        )

        # Ensure dimensions agree.
        length = min(
            len(t),
            len(cpu_mean),
            len(cpu_min),
            len(cpu_max),
        )

        t = t[:length]
        cpu_mean = cpu_mean[:length]
        cpu_min = cpu_min[:length]
        cpu_max = cpu_max[:length]

        ax.plot(
            t,
            cpu_mean,
            linewidth=2.2,
            label="Mean CPU efficiency",
        )

        ax.fill_between(
            t,
            cpu_min,
            cpu_max,
            color = 'red',
            alpha=0.20,
            label="CPU efficiency range",
        )

        add_scenario_edges(
            ax,
            scenarios_edges,
        )

        ax.set_ylim(
            0,
            1,
        )

        mean_std = np.mean(
            res.get(
                "CPU_STD",
                [np.nan],
            )
        )

        ax.set_title(
            f"{res['NAME']}\n"
            f"Mean CPU STD = {mean_std:.3f}"
        )

        ax.set_xlabel(
            "Simulation Time"
        )

        ax.set_ylabel(
            "CPU Efficiency"
        )

        ax.legend(
            loc="best",
            frameon=True,
        )

        ax.margins(
            x=0.02
        )

    for i in range(
        schedulers,
        len(axes),
    ):
        axes[i].remove()

    fig.suptitle(
        "Virtual CPU Efficiency by Scheduler",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.show()


def plot_metrics(
    results_list,
    num_bins=50,
    scenarios_edges=None,
):

    if scenarios_edges is None:
        scenarios_edges = []

    configure_plot_style()


    plot_time_metrics(
        results_list,
        num_bins=num_bins,
        scenarios_edges=scenarios_edges,
    )


    plot_performance_metrics(
        results_list
    )


    plot_sla_violations(
        results_list
    )

    plot_cpu_efficiency(
        results_list,
        scenarios_edges=scenarios_edges,
    )