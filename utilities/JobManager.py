import numpy as np

class JobManager:

    def __init__(self, jobs: list):

        self.jobs = jobs
        self.workload = True
        self.workload_size = len(jobs)

        self.arrived_jobs = []
        self.ready_tasks = []
        self.running_tasks = []
        self.pending_tasks = 0
        self.finished_jobs = []



    def update_arrival_jobs(self, t):

        out_jobs = []

        for job in self.jobs:
            if job.time_arrived <= t and not job.counted:
                job.counted = True
                out_jobs.append(job)

        self.arrived_jobs.extend(out_jobs)
        self.workload_size -= len(out_jobs)

        if self.workload_size <= 0:
            self.workload = False



    def update_ready_tasks(self):

        ready_task = [
            task for task in self.ready_tasks
            if task.status == 1   
        ]

        for job in self.arrived_jobs:
            for task in job.get_ready_tasks():

                if task.status == 1 and task not in ready_task:
                    ready_task.append(task)

        ready_task.sort(key=lambda task: task.arrival_time)

        self.ready_tasks = ready_task


    def update_running_tasks(self):

        running_task = []

        for job in self.arrived_jobs:
            running_task.extend(job.get_running_tasks())

        self.running_tasks = running_task



    def update_finished_jobs(self):

        finished_jobs = []

        for job in self.arrived_jobs:

            if all(
                task.status == 0 and task.end_time is not None
                for task in job.tasks.values()
            ):
                job.success = True
                finished_jobs.append(job)

        for job in finished_jobs:
            self.arrived_jobs.remove(job)
        
        for job in finished_jobs :
                job.end_time = max([tsk.end_time for tsk in job.tasks.values()])

        self.finished_jobs.extend(finished_jobs)



    def initialize(self, t=0):
        self.update_arrival_jobs(t)
        self.update_ready_tasks()