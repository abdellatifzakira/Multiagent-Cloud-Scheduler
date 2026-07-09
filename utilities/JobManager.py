import numpy as np
class JobManager :
    def __init__(self,
                 jobs : list):
        self.jobs = jobs
        self.workload = True
        self.workload_size = len(jobs)
        
    def update_job_list(self,_t) :
        out_jobs = []
        for job in self.jobs :
            if job.time_arrived <= _t and not job.counted:
                out_jobs.append(job)
                job.counted = True
        self.workload_size -= len(out_jobs)
        if self.workload_size <= 0 :
            self.workload = False
        return out_jobs
    
    

    