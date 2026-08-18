# environment/MetricsManager.py

import numpy as np
from collections import Counter

from utilities.helpers import parse_power_equation


class MetricsManager:
    def __init__(
        self,
        server_farm=None,
        jobs=None,
    ):
        self.server_farm = server_farm
        self.jobs = {
            j.id: j
            for j in (jobs or [])
        }

        self.finished_jobs = []
        self.running_tasks = []

        self.servers = (
            self.server_farm.servers.values()
            if self.server_farm is not None
            else []
        )

        self.results = {}

    def set_name(self, name):
        self.results["NAME"] = name

    def initialize(self):
        self.results = {}

        self.results["NAME"] = None
        self.results["TIMELINE"] = []

        self.results["CPU"] = {
            s: []
            for s in self.server_farm.servers.values()
        }

        self.results["CPU_GLOBAL"] = {
            s: []
            for s in self.server_farm.servers.values()
        }

        self.results["RAM"] = {
            s: []
            for s in self.server_farm.servers.values()
        }

        self.results["STORAGE"] = {
            s: []
            for s in self.server_farm.servers.values()
        }

        self.results["QUEUE_LENGTH"] = {
            s: []
            for s in self.server_farm.servers.values()
        }

        self.results["CPU_STD"] = []
        self.results["POWER_PRICE"] = []
        self.results["POWER"] = []
        self.results["DATA_TRANSFER"] = []
        self.results["CUM_DATA_TRANSFER"] = []
        self.results["SLA"] = []
        self.results["SLA_VAR"] = []
        self.results[
            "JOB_MEAN_COMPLETION_TIME"
        ] = []
        self.results["NETWORK_LATENCY"] = []

    def get_sla_violation_rate(self):
        if not self.finished_jobs:
            return 0

        return int(
            np.sum(
                [
                    (
                        job.end_time
                        - job.time_arrived
                    )
                    > job.sla_limit
                    for job in self.finished_jobs
                ]
            )
        )

    def get_mean_completion_time(self):
        if not self.finished_jobs:
            return 0.0

        return float(
            np.mean(
                [
                    job.end_time
                    - job.time_arrived
                    for job in self.finished_jobs
                ]
            )
        )

    def get_network_latency(self):
        latencies = [
            task.time_data_arrival
            - task.start_time
            for job in self.jobs.values()
            for task in job.tasks.values()
            if (
                task.time_data_arrival is not None
                and task.start_time is not None
            )
        ]

        return float(
            np.sum(latencies)
        ) if latencies else 0.0

    def monitor_data_transfer(self):
        data_transfer = 0.0

        for running in self.running_tasks:
            if (
                len(running.parents) > 0
                and not running.monitored
            ):
                for parent in running.parents:
                    if (
                        parent.server is not None
                        and parent.server
                        != running.server
                    ):
                        data_transfer += (
                            self.jobs[
                                running.job_id
                            ]
                            .data_transfer_weights[
                                (parent.id, running.id)
                            ]
                        )

                running.monitored = True

        return float(data_transfer)

    def collect_data(self, job_manager, t):
        self.finished_jobs = (
            job_manager.finished_jobs
        )

        self.running_tasks = (
            job_manager.running_tasks
        )

        self.results["TIMELINE"].append(t)

        for server in (
            self.server_farm.servers.values()
        ):
            self.results["CPU"][server].append(
                server.virtual_cpu_efficiency()
            )

            self.results["CPU_GLOBAL"][
                server
            ].append(
                server.cpu_utilization()
            )

            self.results["RAM"][server].append(
                server.ram_utilization()
            )

            self.results["STORAGE"][
                server
            ].append(
                server.storage_utilization()
            )

            self.results["QUEUE_LENGTH"][
                server
            ].append(
                len(server.task_queue)
            )

        self.results["POWER_PRICE"].append(
            self.server_farm.get_power_price()
        )

        self.results["POWER"].append(
            self.server_farm.get_power()
        )

        transfer = self.monitor_data_transfer()

        self.results["DATA_TRANSFER"].append(
            transfer
        )

        self.results["CUM_DATA_TRANSFER"] = (
            np.cumsum(
                self.results["DATA_TRANSFER"]
            )
        )

        self.results["SLA"].append(
            self.get_sla_violation_rate()
        )

        self.results[
            "JOB_MEAN_COMPLETION_TIME"
        ].append(
            self.get_mean_completion_time()
        )

        if self.results["SLA"]:
            self.results["SLA_VAR"] = np.diff(
                self.results["SLA"],
                prepend=self.results["SLA"][0],
            )

        current_cpu = [
            self.results["CPU"][s][-1]
            for s in self.results["CPU"]
        ]

        self.results["CPU_STD"].append(
            float(np.std(current_cpu))
        )

        self.results["NETWORK_LATENCY"].append(
            self.get_network_latency()
        )

    def print_experience_summary(self):
        if not self.finished_jobs:
            print("NO FINISHED JOBS")
            return

        completion_time = np.asarray(
            [
                job.end_time
                - job.time_arrived
                for job in self.finished_jobs
            ],
            dtype=np.float32,
        )

        makespan = (
            max(
                job.end_time
                for job in self.finished_jobs
            )
            - min(
                job.time_arrived
                for job in self.finished_jobs
            )
        )

        print(
            "MEAN COMPLETION TIME : "
            f"{np.mean(completion_time):.3f}"
        )

        print(
            "95TH PERCENTILE : "
            f"{np.percentile(completion_time, 95):.3f}"
        )

        if self.results["POWER_PRICE"]:
            print(
                "POWER PRICE : "
                f"MEAN = "
                f"{np.mean(self.results['POWER_PRICE']):.3f}, "
                f"RANGE = "
                f"({np.min(self.results['POWER_PRICE']):.3f}, "
                f"{np.max(self.results['POWER_PRICE']):.3f}), "
                f"STD : "
                f"{np.std(self.results['POWER_PRICE']):.3f}"
            )

        if self.results["POWER"]:
            print(
                "POWER CONSUMPTION : "
                f"MEAN = "
                f"{np.mean(self.results['POWER']):.3f}, "
                f"RANGE = "
                f"({np.min(self.results['POWER']):.3f}, "
                f"{np.max(self.results['POWER']):.3f}), "
                f"STD : "
                f"{np.std(self.results['POWER']):.3f}"
            )

        if len(self.results["CUM_DATA_TRANSFER"]) > 0:
            print(
                "CUM DATA TRANSFER : "
                f"{float(np.max(self.results['CUM_DATA_TRANSFER'])):.3f}"
            )

        if self.results["CPU_STD"]:
            print(
                "CPU STD : "
                f"MEAN = "
                f"{np.mean(self.results['CPU_STD']):.3f}, "
                f"RANGE = "
                f"({np.min(self.results['CPU_STD']):.3f}, "
                f"{np.max(self.results['CPU_STD']):.3f})"
            )

        print(
            "FINAL SLA VIOLATION RATE "
            "(IN JOB COUNT) : "
            f"{max(self.results['SLA'])}/"
            f"{len(self.finished_jobs)}"
        )

        if self.results["NETWORK_LATENCY"]:
            print(
                "FINAL NETWORK LATENCY : "
                f"{max(self.results['NETWORK_LATENCY']):.3f}"
            )

        print(
            "FINAL MAKE SPAN : "
            f"{makespan:.3f}"
        )

        print(
            "TIMELINE LENGTH : "
            f"{len(self.results['TIMELINE'])}"
        )

    def print_after_run_check(self):
        all_tasks = [
            task
            for job in self.finished_jobs
            for task in job.tasks.values()
        ]

        states = Counter(
            task.status
            for task in all_tasks
        )

        print("=" * 50)
        print("AFTER RUN CHECK")
        print("=" * 50)

        print(
            f"TOTAL JOBS      : "
            f"{len(self.finished_jobs)}"
        )

        print(
            f"TOTAL TASKS     : "
            f"{len(all_tasks)}"
        )

        print("\nTASK STATUS DISTRIBUTION:")
        print(states)

        print("\nEXPECTED:")
        print("{0: finished tasks}")

        print("\nDETAIL:")

        for status, count in states.items():
            print(
                f"STATUS {status}: {count}"
            )

        print("=" * 50)

    def print_data_integrity_report(self):
        data_transfer = sum(
            sum(
                job.data_transfer_weights.values()
            )
            for job in self.jobs.values()
            if job.data_transfer_weights
            is not None
        )

        sent_data = sum(
            sum(
                data[0]
                for data in server.outgoing_data.values()
            )
            for server in self.servers
        )

        received_data = sum(
            sum(
                data
                for data in server.received_data.values()
            )
            for server in self.servers
        )

        saved_data = sum(
            sum(
                data[0]
                for data in server.saved_data.values()
            )
            for server in self.servers
        )

        print("\nDATA INTEGRITY CHECK")
        print(
            "TOTAL DATA TRANSFER "
            "PARENT-TO-CHILD IN THE WORKLOAD : "
            f"{data_transfer}"
        )

        print(
            f"TOTAL SAVED DATA : {saved_data}"
        )

        print(
            f"TOTAL SENT DATA : {sent_data}"
        )

        print(
            f"TOTAL RECEIVED DATA : {received_data}"
        )

        print(
            "TOTAL EXCHANGE DATA "
            "(SENT + SAVED) : "
            f"{saved_data + sent_data}"
        )

        print(
            ">> LOST DATA "
            "(DUE TO SIM ERROR) : "
            f"{data_transfer - (saved_data + sent_data)}"
        )

        print(
            "TOTAL EXCHANGE DATA "
            "(RECEIVED + SAVED) : "
            f"{saved_data + received_data}"
        )

        print(
            ">> LOST DATA "
            "(DUE TO SIM ERROR) : "
            f"{data_transfer - (saved_data + received_data)}"
        )

    def print_power_model_integrity_check(self):
        if not self.results["POWER"]:
            return

        print("=" * 50)
        print("POWER MODEL INTEGRITY CHECK")
        print(
            f"DYNAMIC POWER MODEL = "
            f"{self.server_farm.power_model}"
        )

        computed_power = np.zeros(
            len(self.results["POWER"])
        )

        for server in self.servers:
            cpu = self.results[
                "CPU_GLOBAL"
            ][server]

            ram = self.results["RAM"][server]
            storage = self.results[
                "STORAGE"
            ][server]

            for i in range(len(cpu)):
                computed_power[i] += (
                    server.power_function(
                        cpu[i],
                        ram[i],
                        storage[i],
                    )
                    + server.static_power
                )

        actual_power = np.asarray(
            self.results["POWER"],
            dtype=np.float64,
        )

        denominator = np.maximum(
            np.abs(actual_power),
            1e-12,
        )

        relative_error = np.max(
            np.abs(
                computed_power - actual_power
            )
            / denominator
        )

        print(
            "FINAL SIMULATION RELATIVE ERROR : "
            f"{relative_error * 100:.6f} %"
        )

    def print_infrastructure_details(self):
        print(
            "<==== INFRASTRUCTURE INSIGHT : ====>"
        )

        for server in (
            self.server_farm.servers.values()
        ):
            print(
                f"SERVER ID {server.id}"
            )

            print(
                f"SERVER CPU {server.c_cpu}"
            )

            print(
                f"SERVER RAM {server.c_ram}"
            )

            print(
                f"SERVER COMPUTE POWER "
                f"{server.compute_power}"
            )

            print(
                f"PEAK CPU USAGE "
                f"{server.peak_cpu}"
            )

            queue_history = self.results[
                "QUEUE_LENGTH"
            ][server]

            if queue_history:
                print(
                    "QUEUE LENGTH : "
                    f"MEAN = {np.mean(queue_history)}, "
                    f"RANGE = "
                    f"[{min(queue_history)},"
                    f"{max(queue_history)}]"
                )

            print("VMS INSIGHT")

            for vm in server.vms.values():
                print(
                    f"=========> VM ID : {vm.id}"
                )

                print(
                    f"=========> VM CPU : {vm.cpu}"
                )

                print(
                    f"=========> VM RAM : {vm.ram}"
                )

                print(
                    "=========> VM COMPUTE POWER : "
                    f"{vm.compute_power}"
                )

                print(
                    "=========> COMPLETED TASKS : "
                    f"{vm.completed_tasks}"
                )

            print(
                f"TOTAL TASKS RUN ON SERVER "
                f"{server.id} : "
                f"{sum(vm.completed_tasks for vm in server.vms.values())}"
            )

        self.print_power_model_integrity_check()

    def get_state_dict(self):
        state = {}

        state["C_CPU"] = {
            s.id: s.c_cpu * s.virtualization_level
            for s in self.server_farm.servers.values()
        }

        state["C_RAM"] = {
            s.id: s.c_ram * s.virtualization_level
            for s in self.server_farm.servers.values()
        }

        state["C_STORAGE"] = {
            s.id: s.storage
            * s.virtualization_level
            for s in self.server_farm.servers.values()
        }

        state["COMPUTE_POWER"] = {
            s.id: s.compute_power
            * s.virtualization_level
            for s in self.server_farm.servers.values()
        }

        state["EFFECTIVE_COMPUTE"] = {
            s.id:
                state["COMPUTE_POWER"][s.id]
                / max(
                    sum(
                        task.cpu
                        for task in s.hosted_tasks.keys()
                    ),
                    1.0,
                )
            for s in self.server_farm.servers.values()
        }

        state["CPU"] = {
            s.id: self.results["CPU"][s]
            for s in self.server_farm.servers.values()
        }

        state["RAM"] = {
            s.id: self.results["RAM"][s]
            for s in self.server_farm.servers.values()
        }

        state["STORAGE"] = {
            s.id: self.results["STORAGE"][s]
            for s in self.server_farm.servers.values()
        }

        state["QUEUE"] = {
            s.id:
                sum(
                    task.cpu
                    for task in s.task_queue
                )
                if s.task_queue
                else 0.0
            for s in self.server_farm.servers.values()
        }

        state["MAX_AVAILABLE_CPU"] = {
            s.id:
                sum(
                    vm.cpu - vm.used_cpu
                    for vm in s.vms.values()
                )
            for s in self.server_farm.servers.values()
        }

        state["MAX_POWER"] = {
            s.id: s.get_max_power_consumption()
            for s in self.server_farm.servers.values()
        }

        state["POWER"] = {
            s.id: s.get_power_consumption()
            for s in self.server_farm.servers.values()
        }

        state["DATA_TRANSFER"] = (
            self.results["DATA_TRANSFER"]
        )

        state["SLA"] = self.results["SLA"]

        state["CPU_STD"] = (
            self.results["CPU_STD"]
        )

        return state

    def compute_reward(
        self,
        action,
        tasks=None,
        previous_server_state=None,
    ):
        action = np.asarray(
            action,
            dtype=np.int64,
        )

        if tasks is None:
            return np.zeros(
                len(action),
                dtype=np.float32,
            )

        rewards = np.zeros(
            len(action),
            dtype=np.float32,
        )

        servers = self.server_farm.servers

        for index, server_id in enumerate(action):
            if server_id not in servers:
                rewards[index] = -1.0
                continue

            task = tasks[index]

            server = servers[server_id]

            capacity_cpu = max(
                float(
                    server.c_cpu
                    * server.virtualization_level
                ),
                1e-6,
            )

            capacity_ram = max(
                float(
                    server.c_ram
                    * server.virtualization_level
                ),
                1e-6,
            )

            available_cpu = max(
                float(
                    sum(
                        vm.cpu - vm.used_cpu
                        for vm in server.vms.values()
                    )
                ),
                0.0,
            )

            available_ram = max(
                float(
                    sum(
                        vm.ram - vm.used_ram
                        for vm in server.vms.values()
                    )
                ),
                0.0,
            )

            task_cpu = max(
                float(task.cpu),
                1e-6,
            )

            task_ram = max(
                float(task.ram),
                1e-6,
            )

            cpu_fit = min(
                available_cpu / task_cpu,
                1.0,
            )

            ram_fit = min(
                available_ram / task_ram,
                1.0,
            )

            if (
                available_cpu < task_cpu
                or available_ram < task_ram
            ):
                rewards[index] = -1.0
                continue

            projected_queue = (
                sum(
                    queued.cpu
                    for queued in server.task_queue
                )
                + task_cpu
            ) / capacity_cpu

            projected_utilization = min(
                1.0,
                1.0
                - (
                    available_cpu
                    - task_cpu
                )
                / capacity_cpu,
            )

            current_power = float(
                server.get_power_consumption()
            )

            max_power = max(
                float(
                    server.get_max_power_consumption()
                ),
                1e-6,
            )

            power_ratio = np.clip(
                current_power / max_power,
                0.0,
                1.0,
            )

            effective_compute = max(
                float(
                    self.get_state_dict()[
                        "EFFECTIVE_COMPUTE"
                    ][server_id]
                ),
                1e-9,
            )

            instruction_scale = max(
                float(
                    getattr(
                        task,
                        "num_instructions",
                        1.0,
                    )
                ),
                1.0,
            )

            expected_load = (
                instruction_scale
                / effective_compute
            )

            relative_load = np.tanh(
                expected_load / 10.0
            )

            balance_penalty = (
                abs(
                    projected_utilization
                    - np.mean(
                        [
                            (
                                1.0
                                - sum(
                                    vm.cpu
                                    - vm.used_cpu
                                    for vm in s.vms.values()
                                )
                                / max(
                                    float(
                                        s.c_cpu
                                        * s.virtualization_level
                                    ),
                                    1e-6,
                                )
                            )
                            for s in servers.values()
                        ]
                    )
                )
            )

            reward = (
                - 0.05 * projected_queue
                - 0.10 * power_ratio
                - 0.15 * relative_load
                - 0.35 * balance_penalty
            )

            rewards[index] = float(
                np.clip(
                    reward,
                    -2.0,
                    2.0,
                )
            )

        return rewards

    def clamp_results(self, critical_time):
        self.results["TIMELINE"] = [
            t
            for t in self.results["TIMELINE"]
            if t <= critical_time
        ]

        critical_length = len(
            self.results["TIMELINE"]
        )

        for server in (
            self.server_farm.servers.values()
        ):
            for key in (
                "CPU",
                "CPU_GLOBAL",
                "RAM",
                "STORAGE",
                "QUEUE_LENGTH",
            ):
                self.results[key][server] = (
                    self.results[key][server][
                        :critical_length
                    ]
                )

        for key in (
            "CPU_STD",
            "POWER_PRICE",
            "POWER",
            "DATA_TRANSFER",
            "CUM_DATA_TRANSFER",
            "SLA",
            "SLA_VAR",
            "JOB_MEAN_COMPLETION_TIME",
            "NETWORK_LATENCY",
        ):
            self.results[key] = (
                self.results[key][
                    :critical_length
                ]
            )