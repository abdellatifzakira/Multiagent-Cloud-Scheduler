
import copy
import numpy as np

from utilities.DAG_handlers import *
from classes.JobClass import generate_workload
from baselines.RoundRobin import RoundRobinScheduler
from comparison.LeastLoaded import LeastLoadedScheduler
from comparison.LeastCommunication import DataLocalityAwareScheduler
from comparison.LeastEnergy import EnergyAwareScheduler
from utilities.helpers import *
from ExperimentRunner import Experiment

import time

from RL.agents.RandomAgent import RandomAgent
from RL.agents.SequentialDQNAgent import SequentialDQNAgent
from RL.agents.DQNAgent import DQNAgent
from classes.ServerFarmClass import build_random_server_farms

def run_sensitivity(
    infrastructure,
    schedulers,
    seeds,
    workload_generator,
):

    results = {
        scheduler.name: {
            "POWER": [],
            "CPU_STD": [],
            "COMPLETION_TIME": [],
            "MAKESPAN": [],
        }
        for scheduler in schedulers
    }

    for seed in seeds:

        print("=" * 60)
        print(f"SENSITIVITY TEST SEED : {seed}")
        print("=" * 60)

        jobs_test, _ = workload_generator(
            seed=seed
        )

        for scheduler in schedulers:

            print(
                f"\nRunning {scheduler.name} "
                f"with test seed {seed}"
            )

            # Each scheduler receives its own copy of the
            # infrastructure and workload through Experiment.
            exp = Experiment(
                infrastructure=copy.deepcopy(
                    infrastructure
                ),
                schedulers=[scheduler],
                evaluation=copy.deepcopy(
                    jobs_test
                ),
                scenarios_edges=[],
                time_step=0.005,
                network_overhead_enabled=True,
                power_model="DEFAULT",
                episodes=0,
                batch_size=25,
                clamp_results=False,
                verbose=False,
            )

            exp.build_environment()
            exp.run_experiment()

            env = exp.environments[scheduler]

            summary = (
                env.metrics_manager
                .print_experience_summary()
            )

            if summary is None:
                raise RuntimeError(
                    f"No summary returned for "
                    f"{scheduler.name} "
                    f"with seed {seed}."
                )

            (
                avg_power,
                avg_cpu_std,
                avg_completion_time,
                makespan,
            ) = summary

            results[scheduler.name]["POWER"].append(
                avg_power
            )

            results[scheduler.name]["CPU_STD"].append(
                avg_cpu_std
            )

            results[scheduler.name][
                "COMPLETION_TIME"
            ].append(
                avg_completion_time
            )

            results[scheduler.name]["MAKESPAN"].append(
                makespan
            )

    summary = _compute_statistics(results)

    _print_summary(summary)

    return results, summary


def _compute_statistics(results):
    """
    Compute mean and standard deviation across test seeds.
    """

    summary = {}

    for scheduler_name, metrics in results.items():

        summary[scheduler_name] = {}

        for metric, values in metrics.items():

            values = np.asarray(
                values,
                dtype=np.float64
            )

            summary[scheduler_name][metric] = {
                "MEAN": float(np.mean(values)),
                "STD": float(np.std(values)),
            }

    return summary


def _print_summary(summary):
    """
    Print mean and standard deviation for each scheduler.
    """

    print("\n")
    print("=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)

    for scheduler_name, metrics in summary.items():

        print(f"\n{scheduler_name}")

        for metric, values in metrics.items():

            print(
                f"{metric:20s} : "
                f"MEAN = {values['MEAN']:.4f}, "
                f"STD = {values['STD']:.4f}"
            )

    print("=" * 70)
    



def generate_test_workload(seed):
    return generate_workload(
        seed=seed,
        num_jobs=[100, 100, 100],
        mean_job_gap=[0.1, 0.05, 0.01],
        num_tasks_per_job=5,
        cpu_req_per_task=[12, 64],
        ram_req_per_task=[2, 32],
        task_size=[32, 64],
        edge_probability=0.0,
        instructions_per_task=[
            250e5,
            500e5,
        ],
        data_transfer_range=[
            512,
            1024,
        ],
    )



infra_seed = 13
schedulers_seed = 42

    
    
server_farm = build_random_server_farms(
        cpu_range=[1024, 1024 * 2],
        ram_range=[4096, 4096 * 4],
        storage_range=[20480, 20480 * 2],
        compute_power_range=[
            1e9,
            2e9,
        ],
        max_vms_count=5,
        alphas=[100, 500],
        betas=[2, 5],
        server_count=4,
        virtual_allocation=[
            0.9,
            0.95,
        ],
        bandwidth=[
            4096.0,
            16384.0,
        ],
        mode="EXECUTION_TIME",
        seed=infra_seed,
    )









results, summary = run_sensitivity(
    infrastructure=server_farm,
    schedulers=[DQNAgent(
                    epsilon=1.0,
                    epsilon_decay=0.995,
                    seed=schedulers_seed,
                    buffer_capacity=10_000,
                    model_path = "models/No_DAG_DQN.pth",
                    resume_training=False),
                SequentialDQNAgent(
                    epsilon=1.0,
                    epsilon_decay=0.995,
                    seed=schedulers_seed,
                    buffer_capacity=10_000,
                    model_path = "models/No_DAG_SDQN.pth",
                    resume_training=False),
                LeastLoadedScheduler(mode='CPU'),
                RoundRobinScheduler(),
                EnergyAwareScheduler(),
                RandomAgent(seed=schedulers_seed)    ],
    seeds=[12, 224, 31, 45, 56, 55, 75, 30, 20, 11],
    workload_generator=generate_test_workload,
)
