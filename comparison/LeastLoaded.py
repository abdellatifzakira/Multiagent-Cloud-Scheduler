import numpy as np
import random
import math
class LeastLoadedScheduler:
    def __init__(self,
                 server_farm = None,
                 data_transfer_manager = None,
                 job_manager = None,
                 mode : str = 'CPU',
                 sorting : str = 'FIFO'
                 ):
        self.server_farm = server_farm
        self.job_manager = job_manager
        self.jobs = self.job_manager.update_job_list(0)
        self.job_dict = self.populate_job(self.jobs)
        self.data_transfer_manager = data_transfer_manager
        self.servers = self.populate_servers()
        self.ready_tasks = self.find_entry_tasks()
        self.pointer = 0
        self.running_tasks = []
        self.mode = mode
        self.sorting = sorting
        self.pending_tasks = 0
        
        assert mode in ['CPU', 'RAM', 'QUEUE'], "[INVALID MODE]\nAVAILABLE MODES : CPU | RAM | QUEUE "
        assert sorting in ['FIFO', 'CPU', 'RUNTIME'], "[INVALID INPUT]\nAVAILABLE SORTING PARAMETERS : FIFO | CPU | RUNTIME"


    
    def find_entry_tasks(self):
        ready_tasks = []
        for job in self.jobs :
            ready_tasks.extend(job.get_entry_points())
        for task in ready_tasks:
            task.status = 1
        return ready_tasks

    def find_ready_tasks(self, _t):

        # keep old unscheduled tasks
        ready_task = [
            task for task in self.ready_tasks
            if task.status == 1
        ]
        new_jobs = self.job_manager.update_job_list(_t)
        
        # add new jobs to the dictionary 
        for job in new_jobs:
            self.job_dict[job.id] = job
        
        for job in new_jobs:
            self.jobs.append(job)
            ready_task.extend(job.get_entry_points())

        # check active jobs for newly unlocked tasks
        for job in self.jobs:
            for task in job.get_ready_tasks():
                if task not in ready_task:
                    ready_task.append(task)
        
         # sort by arrival time
        match self.sorting :
            case 'FIFO' :
                ready_task.sort(
                    key=lambda task: task.arrival_time
                )
            case 'CPU' :
                ready_task.sort(
                    key=lambda task: task.cpu
                )
            case 'RUNTIME' :
                ready_task.sort(
                    key=lambda task: task.runtime
                )
        return ready_task
    
    def find_running_tasks(self):
        ready_task = []
        for job in self.jobs :
            ready_task.extend(job.get_running_tasks())
        return ready_task
        
    def populate_servers(self):
        return self.server_farm.servers
    
    def populate_job(self, jobs) :
        return {job.id : job for job in jobs}
    
    def get_finished_jobs(self):
        finished_jobs = []

        for job in self.jobs:

            if all(
                task.status == 0 and task.end_time is not None
                for task in job.tasks.values()
            ):
                finished_jobs.append(job)

        return finished_jobs
            
    
    def get_sla_violation_rate(self) :
        if self.get_finished_jobs() != []:
            for job in self.get_finished_jobs() :
                job.end_time = max([tsk.end_time for tsk in job.tasks.values()])
                if job.end_time is None :
                    print("END TIME NOT SET")
                    exit(-1)
            sla_violation_rate = (
                        np.sum([
                            (job.end_time - job.time_arrived) > job.sla_limit
                            for job in self.get_finished_jobs()
                        ])
                        / len(self.get_finished_jobs())
                    ) * 100
            return sla_violation_rate
        else :
            return 0
        
    
                    
    def monitor_data_transfer(self):
        data_transfer= 0
        for running in self.running_tasks :
            if len(running.parents)>0 and not running.monitored:
                for parent in running.parents :
                    if parent.server != running.server :
                        data_transfer += self.job_dict[running.job_id].data_transfer_weights[(parent.id,running.id)]
                running.monitored = True
        return data_transfer
    
    def get_least_busy_server(self):
        servers = list(self.server_farm.servers.values())

        match self.mode:
            case 'QUEUE' :
                """
                the waiting time in the queue + the remaining execution time
                of the hosted asks all divided by the parallelism capacity of the server
                """
                loads = [
                    (s, (sum(tsk.runtime for tsk in s.task_queue) + sum(tsk.timer for tsk in s.hosted_tasks.keys()))/sum(vm.max_concurrent_tasks for vm in s.vms.values())
                     )
                    for s in servers
                ]
            case 'CPU' :
                """
                a score based on the actual cpu usage,
                max vm cpu availability, and queue cpu req pressure
                """
                loads = []
                for s in servers:
                    current = s.cpu_utilization()

                    max_available_vm_cpu = max(
                        vm.cpu - vm.used_cpu
                        for vm in s.vms.values()
                    )
                    
                    # A weighted queue pressure, assuming front tasks are more likely to get assigned,
                    # therfore more likely to surge the vm/server
                    queue_pressure = np.sum(
                        [tsk.cpu*math.exp(-0.3*i) for i, tsk in enumerate(s.task_queue)]
                    )
                    score = current + max(
                        0,
                        queue_pressure - max_available_vm_cpu
                    )

                    loads.append((s, score))

            case 'RAM' :
                loads = [(s, s.ram_utilization()) for s in servers]

        min_load = min(loads, key=lambda x: x[1])[1]

        # To prevent floating point errors
        candidates = [ s for s, l in loads
                    if abs(l - min_load) < 1e-9 ]

        return random.choice(candidates)
        
        
        
    def schedule(self):
        n = len(self.servers)
        _t = 0
        time_line = []
        cpu_usage = {s  : [] for s in self.server_farm.servers.values()}
        power_price = []
        data_transfer = []
        sla_violation = []
        sla_violation_var = []
        workload_std = []
        server_schedules = {s : [] for s in self.server_farm.servers.values()}
        while self.job_manager.workload or len(self.running_tasks)>0 or len(self.ready_tasks)>0 or self.pending_tasks > 0 :
            self.pending_tasks = 0
            for task in self.ready_tasks :
                    server = self.get_least_busy_server()
                    server.add_task_to_queue(task, _t)
                    
            for server in self.server_farm.servers.values() :
                #server_schedules[server].append(len(list(server.task_queue))) 
                server_schedules[server].append(server.get_storage_usage())
                server.execute_tasks(_t)
                self.pending_tasks += len(list(server.task_queue))
                
            for s in cpu_usage.keys() :
                cpu_usage[s].append(s.cpu_utilization())
            
            
            power_price.append(self.server_farm.get_power_price())
            self.ready_tasks = self.find_ready_tasks(_t = _t)
            self.running_tasks = self.find_running_tasks()
            data_transfer.append(self.monitor_data_transfer())
            sla_violation.append(self.get_sla_violation_rate())
            sla_violation_var = np.diff(sla_violation, prepend=sla_violation[0])
            workload_std.append(np.std([cpu_usage[s][_t] for s in cpu_usage.keys()]))
            time_line.append(_t)
            self.server_farm.update_farm_state(t=_t)
            _t += 1

        
        
        return  (time_line,
                cpu_usage,
                power_price,
                server_schedules,
                data_transfer,
                sla_violation,
                sla_violation_var,
                workload_std
                )


