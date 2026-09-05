# RL/CloudEnv.py

import copy

import gymnasium as gym
import numpy as np

from utilities.JobManager import JobManager
from environment.MetricsManager import MetricsManager
from environment.NetworkManager import NetworkManager



class CloudEnv(gym.Env):
    def __init__(
        self,
        server_farm,
        jobs,
        time_step=0.01,
        network_overhead=False,
        evaluation=None,
        agent=None,
        batch_size = 4,
        clamp_results = False,
    ):
        super().__init__()

        self.server_farm = None
        self.job_manager = None
        self.metrics_manager = None
        self.network_manager = None

        self.agent = agent
        self.clamp_results = clamp_results

        self.network_overhead = (
            network_overhead
        )

        self.time_step = time_step
        self.current_time = 0.0

        self.num_servers = len(
            server_farm.servers
        )

        self.batch_size = batch_size

        self.action_space = gym.spaces.MultiDiscrete(
            [self.num_servers] * self.batch_size
        )

        self.agent.action_space = (
            self.action_space
        )

        self.agent.batch_size = (
            self.batch_size
        )

        self.agent.num_servers = (
            self.num_servers
        )

        self.agent.build()

        self.ready_tasks = []
        self.results = {}

        self.reward_buffer = []
        self.draining_time = None

        self.mode = "TRAIN"

        self.server_farm_backup = (
            server_farm
        )

        self.jobs = jobs
        self.evaluation = evaluation

        self.current_batch = []

        self._decision_tasks = []
        self._decision_actions = None
        
        self.reward_history = []
        
        self.last_agent_action = 0.0

    def is_running(self):
        """
        print(
                    self.job_manager.workload
                    , len(
                        self.job_manager.running_tasks
                    )
                    ,
                     len(
                        self.job_manager.ready_tasks
                    )
                    ,
                    self.job_manager.pending_tasks
                    
                )
        """
        return (
            self.job_manager.workload
            or len(
                self.job_manager.running_tasks
            )
            > 0
            or len(
                self.job_manager.ready_tasks
            )
            > 0
            or self.job_manager.pending_tasks
            > 0
        )

    def reset(
        self,
        seed=None,
        options=None,
        episode=0,
    ):
        super().reset(seed=seed)

        self.draining_time = None

        farm = copy.deepcopy(
            self.server_farm_backup
        )

        if (
            self.mode == "TEST"
            and self.evaluation is not None
        ):
            jobs = copy.deepcopy(
                self.evaluation
            )
        else:
            jobs = copy.deepcopy(
                self.jobs
            )

        self.server_farm = farm

        assert all(
            task.status not in {0, 2}
            for job in jobs
            for task in job.tasks.values()
        )

        self.metrics_manager = (
            MetricsManager(
                server_farm=farm,
                jobs=jobs,
            )
        )

        self.job_manager = JobManager(
            jobs=jobs
        )

        self.network_manager = (
            NetworkManager(farm)
        )

        self.server_farm.communication_enabled = (
            self.network_overhead
        )

        self.server_farm.set_communication_mode()
        self.job_manager.initialize(t=0)

        self.reward_buffer = []
        self.current_time = 0.0

        self.current_batch = []
        self._decision_tasks = []
        self._decision_actions = None

        self.metrics_manager.initialize()
        self.ready_tasks = (
            self.job_manager.update_ready_tasks()
        )

        self.current_batch = (
            self.ready_tasks[
                : self.batch_size
            ]
        )

        self.ready_tasks_dict = {}
        
        self.last_agent_action = 0.0

        self.metrics_manager.set_name(
            self.agent.name
        )

        return self.get_state(), {}

    def build_task_dict(self):
        self.ready_tasks_dict = {
            (task.id, task.job_id): task
            for task in self.ready_tasks
        }
    def step(self, action):
            self.last_agent_action = self.current_time

            action = np.asarray(action, dtype=np.int64)

            if all(task.status == 1 for task in self.current_batch):
                self._decision_tasks = list(self.current_batch)
                self._decision_actions = action[: self.batch_size].copy()

                for index, task in enumerate(self.current_batch):
                    server_id = int(self._decision_actions[index])
                    if server_id in self.server_farm.servers:
                        server = self.server_farm.servers[server_id]
                        if server.first_check(task):
                            server.add_task_to_queue(task, t=self.current_time)
            return self._advance_one_tick()

    def _advance_one_tick(self):
        t = self.current_time
        self.job_manager.pending_tasks = 0

        for server in self.server_farm.servers.values():
            server.execute_tasks(t)
        for server in self.server_farm.servers.values():
            self.job_manager.pending_tasks += len(list(server.task_queue))

        self.job_manager.update_running_tasks()
        self.job_manager.update_finished_jobs()
        self.current_time += self.time_step
        self.server_farm.update_farm_state(
            t=self.current_time,
            time_step=self.time_step,
        )

        self.metrics_manager.collect_data(
            self.job_manager,
            self.current_time,
        )

        if self.network_overhead:
            submitted_data = self.server_farm.submit_packets()
            if submitted_data:
                self.network_manager.update_data_packets(submitted_data)
                self.network_manager.resolve_routing()

            self.network_manager.distribute_data_payloads(
                self.current_time,
                self.time_step,
            )

        self.job_manager.update_arrival_jobs(self.current_time)
        self.ready_tasks = self.job_manager.update_ready_tasks()
        self.build_task_dict()

        done = not self.is_running()
        return done
   
    def get_state(self):
        if self.is_scheduling_time():
            self.current_batch = (
                self.ready_tasks[
                    : self.batch_size
                ]
            )

        state = self.metrics_manager.get_state_dict()

        state["TASKS"] = {}

        for task in self.current_batch:
            task_id = (
                task.id,
                task.job_id,
            )

            state["TASKS"][task_id] = {
                "R_CPU": task.cpu,
                "R_RAM": task.ram,
                "SIZE": task.size / 1024,
                "DATA": (
                    sum(
                        task.parent_weights.values()
                    )
                    / 1e6
                ),
                "INSTRUCTIONS": (
                    task.num_instructions / 1e9
                ),
                "PARENTS": {
                    (
                        parent.id,
                        parent.job_id,
                    ): (
                        parent.server.id
                        if parent.server is not None
                        else -1,
                        task.parent_weights[
                            parent
                        ],
                    )
                    for parent in task.parents
                },
            }

        return state

    def get_reward(self, action):
        tasks = list(
            self._decision_tasks
        )

        if not tasks:
            tasks = list(
                self.current_batch
            )

        reward = (
            self.metrics_manager.compute_reward(
                action,
                tasks=tasks,
            )
        )

        self.reward_buffer.append(
            float(np.mean(reward))
            if len(reward)
            else 0.0
        )

        return reward

    def is_scheduling_time(self):
        return (
            len(self.ready_tasks)
            > 0 #self.batch_size
        )

    def is_batch_finished(self):
        return (
            all(
                task.status == 0
                for task in self.current_batch
            )
            if self.current_batch
            else True
        )

    def step_ahead(self):
        return self._advance_one_tick()

    def log(self):
        print("=" * 50)
        print(
            "SUMMARY RUN FOR SCHEDULER : "
        )
        print(self.agent.full_name)
        print("=" * 50)

        self.metrics_manager.print_experience_summary()
        self.metrics_manager.print_after_run_check()
        self.metrics_manager.print_infrastructure_details()

        if self.network_overhead:
            self.metrics_manager.print_data_integrity_report()
        else:
            print(
                "[INFO] : the data integrity report "
                "is not available when communication "
                "overhead is disabled"
            )

        print("=" * 50)
        print(
            "END SUMMARY RUN FOR SCHEDULER : ",
            self.agent.name,
        )
        print("=" * 50)

    def get_results(self):
        if self.agent.trainable :
            self.metrics_manager.results['REWARD'] = self.reward_history
        if self.clamp_results:
            self.metrics_manager.clamp_results(
                critical_time=self.last_agent_action
            )

        return self.metrics_manager.results

    def is_updating_time(self):
        return False