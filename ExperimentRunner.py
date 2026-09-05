# ExperimentRunner.py

from utilities.JobManager import JobManager
from environment.EnvironmentClass import Environment
from environment.MetricsManager import MetricsManager
from environment.NetworkManager import NetworkManager
from utilities.ResultPlotter import plot_metrics
from templates.Scheduler import Scheduler
from RL.agents.Agent import Agent
from RL.CloudEnv import CloudEnv

import copy
import numpy as np
import time


class Experiment:
    def __init__(
        self,
        infrastructure=None,
        jobs=None,
        scenarios_edges=None,
        schedulers=None,
        time_step=0.01,
        network_overhead_enabled=False,
        power_model="DEFAULT",
        evaluation=None,
        episodes=10,
        batch_size = 4,
        clamp_results = False,
        verbose = False
    ):
        self.infrastructure = infrastructure
        self.jobs = jobs or []
        self.scenarios_edges = (
            scenarios_edges or []
        )
        self.schedulers = schedulers or []
        self.clamp_results = clamp_results

        self.time_step = time_step
        self.power_model = power_model
        self.batch_size = batch_size

        self.environments = {}
        self.results = {}

        self.network_overhead_enabled = (
            network_overhead_enabled
        )

        self.evaluation = evaluation
        self.episodes = int(episodes)
        
        self.verbose = verbose

        if not self.network_overhead_enabled:
            print(
                "\n[INFO] : Network communication "
                "overhead is disabled\n"
                "The experiment is under the assumption "
                "of infinite bandwidth\n"
            )
        else:
            print(
                "\n[INFO] : Network communication "
                "overhead is enabled\n"
            )

    def build_environment(self):
        self.environments = {}

        for scheduler in self.schedulers:
            farm = copy.deepcopy(
                self.infrastructure
            )

            farm.set_power_model(
                self.power_model
            )

            job_copy = copy.deepcopy(
                self.evaluation
                if self.evaluation is not None
                else self.jobs
            )

            if isinstance(
                scheduler,
                Scheduler,
            ):
                self.environments[
                    scheduler
                ] = Environment(
                    server_farm=farm,
                    metrics_manager=MetricsManager(
                        farm,
                        jobs=job_copy,
                    ),
                    job_manager=JobManager(
                        jobs=job_copy
                    ),
                    scheduler=scheduler,
                    time_step=self.time_step,
                    network_manager=NetworkManager(
                        server_farm=farm
                    ),
                    network_overhead=(
                        self.network_overhead_enabled
                    ),
                    batch_size=self.batch_size,
                    clamp_results=self.clamp_results
                )

            elif isinstance(
                scheduler,
                Agent,
            ):
                self.environments[
                    scheduler
                ] = CloudEnv(
                    jobs=self.jobs,
                    server_farm=farm,
                    agent=scheduler,
                    network_overhead=(
                        self.network_overhead_enabled
                    ),
                    time_step=self.time_step,
                    evaluation=job_copy,
                    batch_size=self.batch_size,
                    clamp_results=self.clamp_results
                )

    def run_experiment(self):
        for scheduler in self.schedulers:
            print(
                f"{scheduler.name} : "
                "SCHEDULING - STARTS"
            )

            if isinstance(
                scheduler,
                Scheduler,
            ):
                self.results[
                    scheduler
                ] = self.environments[
                    scheduler
                ].run()

                continue

            if not isinstance(
                scheduler,
                Agent,
            ):
                continue

            env = self.environments[scheduler]

            if scheduler.trainable:
                total_episodes = self.episodes
            else:
                total_episodes = 0

            episode = 0
            if getattr(scheduler, "model_path", False):
                scheduler.load_model(scheduler.model_path)
            else :
                setattr(scheduler, "resume_training", True)
            start = time.time()
            while episode <= total_episodes:
                try:
                    if scheduler.trainable:
                        if episode == total_episodes or (scheduler.loaded and not scheduler.resume_training) :
                            env.mode = "TEST"
                            scheduler.epsilon = 0.0
                            episode = total_episodes
                        elif episode > 0:
                            scheduler.decay_epsilon()
                    else:
                        env.mode = "TEST"

                    env.reset(
                        episode=episode
                    )

                    done = False

                    while not done:
                        if env.is_scheduling_time():
                            state = env.get_state()

                            scheduler.observe(
                                state
                            )

                            action = (
                                scheduler.take_action()
                            )

                            done = env.step(
                                action
                            )

                            while (
                                not env.is_scheduling_time()
                                and not done
                            ):
                                done = env.step_ahead()

                            next_state = (
                                env.get_state()
                            )

                            reward = (
                                env.get_reward(
                                    action
                                )
                            )

                            if (
                                scheduler.trainable
                                and env.mode
                                == "TRAIN"
                            ):
                                scheduler.update(
                                    action,
                                    reward,
                                    next_state,
                                    done,
                                )
                        else:
                            done = (
                                env.step_ahead()
                            )

                    if scheduler.trainable:
                        rewards = (
                            env.reward_buffer
                        )

                        mean_reward = (
                            float(
                                np.mean(rewards)
                            )
                            if rewards
                            else 0.0
                        )

                        total_reward = (
                            float(
                                np.sum(rewards)
                            )
                            if rewards
                            else 0.0
                        )
                        
                        env.reward_history.append(total_reward)
                        if self.verbose and (episode % 50 == 0) :
                            print(
                                f"{env.mode = }, "
                                f"{episode = }/"
                                f"{self.episodes}, "
                                f"TOTAL REWARD : "
                                f"{total_reward:.5f}, "
                                f"MEAN REWARD : "
                                f"{mean_reward:.5f},\n"
                                f"LENGTH REWARD : "
                                f"{len(rewards)}, "
                                f"EPSILON : "
                                f"{scheduler.epsilon:.5f}, "
                                f"BUFFER LENGTH : "
                                f"{len(scheduler.buffer)}"
                            )

                    episode += 1

                except KeyboardInterrupt:
                    if scheduler.trainable:
                        print(
                            "TRAINING INTERRUPTED — "
                            "TESTING THE LAST TRAINED MODEL"
                        )

                        episode = total_episodes
                    else :
                        pass

            self.results[
                scheduler
            ] = env.get_results()
            if scheduler.trainable and scheduler.resume_training:
                print(f"Model Trained for : {time.time() - start} s")
                file = f"DAG_{scheduler.name}" if self.network_overhead_enabled else f"No_DAG_{scheduler.name}"
                scheduler.save_model(f"models/{file}.pth")

            env.log()

    def plot_results(self):
        results = list(
            self.results.values()
        )

        plot_metrics(
            results_list=results,
            scenarios_edges=self.scenarios_edges,
        )