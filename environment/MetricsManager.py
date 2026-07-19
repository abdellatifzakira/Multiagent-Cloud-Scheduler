import numpy as np

class MetricsManager:
    def __init__(self,
                 server_farm = None,
                 jobs: list = []

                 ):
        self.server_farm = server_farm
        self.jobs = {j.id : j for j in jobs}
        self.finished_jobs = []
        self.running_tasks = []
        self.servers = self.server_farm.servers.values()
        self.results = {}
        
    def set_name(self, name) :
        self.results['NAME'] = name
        
        
    def initialize(self):
        self.results = {}
        self.results['NAME'] = None
        self.results['TIMELINE'] = []
        self.results['CPU'] = {s  : [] for s in self.server_farm.servers.values()}
        self.results['RAM'] =  {s  : [] for s in self.server_farm.servers.values()}
        self.results['STORAGE'] =  {s  : [] for s in self.server_farm.servers.values()}
        self.results['CPU_STD'] = []
        self.results['POWER_PRICE'] = []
        self.results['DATA_TRANSFER'] = []
        self.results['CUM_DATA_TRANSFER'] = []
        self.results['SLA'] = []
        self.results['SLA_VAR'] = []
        
        
    def get_sla_violation_rate(self) :
        if self.finished_jobs != []:
            sla_violation_rate = (
                        np.sum([
                            (job.end_time - job.time_arrived) > job.sla_limit
                            for job in self.finished_jobs
                        ]))
            
            
            
            return sla_violation_rate
        else :
            return 0
                    
    def monitor_data_transfer(self):
        data_transfer= 0
        for running in self.running_tasks :
            if len(running.parents)>0 and not running.monitored:
                for parent in running.parents :
                    if parent.server != running.server :
                        data_transfer += self.jobs[running.job_id].data_transfer_weights[(parent.id,running.id)]
                running.monitored = True
        return data_transfer
        
    def collect_data(self, job_manager , t):
        
        self.finished_jobs = job_manager.finished_jobs
        self.running_tasks = job_manager.running_tasks
    
        self.results['TIMELINE'].append(t)
        for server in self.server_farm.servers.values() :
                self.results['CPU'][server].append(server.virtual_cpu_efficiency())
                self.results['RAM'][server].append(server.ram_utilization())
                self.results['STORAGE'][server].append(server.storage_utilization())

            
            
        self.results['POWER_PRICE'].append(self.server_farm.get_power_price())
        self.results['DATA_TRANSFER'].append(self.monitor_data_transfer())
        self.results['CUM_DATA_TRANSFER'] = np.cumsum(self.results['DATA_TRANSFER'])
        self.results['SLA'].append(self.get_sla_violation_rate())
        self.results['SLA_VAR'] = np.diff(self.results['SLA'], prepend=self.results['SLA'][0])
        self.results['CPU_STD'].append(np.std([self.results['CPU'][s][t] for s in self.results['CPU'].keys()]))
        
        
        
    
    def print_experience_summary(self) :
        
        print(f"POWER PRICE : MEAN = {np.mean(self.results['POWER_PRICE'])}, RANGE = {np.max(self.results['POWER_PRICE']), np.min(self.results['POWER_PRICE'])}" )
        print(f"DATA TRANSFER : MEAN = {np.mean(self.results['DATA_TRANSFER'])}, RANGE = {np.max(self.results['DATA_TRANSFER']), np.min(self.results['DATA_TRANSFER'])}" )
        print(f"CUM DATA TRANSFER : RANGE = {np.max(self.results['CUM_DATA_TRANSFER']), np.min(self.results['CUM_DATA_TRANSFER'])}" )
        print(f"CPU STD : MEAN = {np.mean(self.results['CPU_STD'])}, RANGE = {np.max(self.results['CPU_STD']), np.min(self.results['CPU_STD'])}" )
        print(f"FINAL SLA VIOLATION RATE (IN JOB COUNT) : {max(self.results['SLA'])}/{len(self.finished_jobs)}")
        print(f"TIMELINE LENGTH : {len(self.results['TIMELINE'])}")
        

