import numpy as np
import time
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
        self.index = 0



    def update_arrival_jobs(self, t):
        out_jobs = []
        while self.jobs[self.index] and self.jobs[self.index].time_arrived <= t and self.workload:
                self.jobs[self.index].counted = True
                out_jobs.append(self.jobs[self.index])
                self.workload_size -= 1
                if self.workload_size <= 0:
                    self.workload = False
                    break
                self.index +=1

        self.arrived_jobs.extend(out_jobs)
       





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
        return self.ready_tasks
    


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
        
    
    def reset(self):
        self.workload_size = len(self.jobs)
        self.workload = True
        self.arrived_jobs = []
        self.ready_tasks = []
        self.running_tasks = []
        self.pending_tasks = 0
        self.finished_jobs = []
        for job in self.jobs :
            job.end_time = None
            job.sla_violated = None,
            job.counted = False
            job.success = False
            for task in job.tasks.values() :
                   task.reset()