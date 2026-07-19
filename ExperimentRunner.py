
from utilities.JobManager import JobManager
from environment.EnvironmentClass import Environment
from environment.MetricsManager import MetricsManager
from utilities.ResultPlotter import plot_metrics
import copy

class Experiment:
        def __init__(self,
                     infrastructure =  None,
                     jobs = [],
                     scenarios_edges = [],
                     schedulers = [],
                     
                ):
            self.infrastructure = infrastructure
            self.jobs = jobs
            self.scenarios_edges = scenarios_edges
            self.schedulers = schedulers
            
            self.environments = {}
            
            self.results = {}
        
            
                
        def build_environment(self):
            
            for scheduler in self.schedulers :

                farm = copy.deepcopy(self.infrastructure)

                job_copy = copy.deepcopy(self.jobs)

                self.environments[scheduler] = Environment(
                                                        server_farm=farm,
                                                        metrics_manager=MetricsManager(
                                                            farm,
                                                            jobs=job_copy
                                                        ),
                                                        job_manager=JobManager(
                                                            jobs=job_copy
                                                        ),
                                                        scheduler=scheduler
                                                    )
        
        def run_experiment(self):
            for scheduler in self.schedulers :
                print(f"{scheduler.name} : SCHEDULING - STARTS")
                self.results[scheduler] = self.environments[scheduler].run()
        
        
        def plot_results(self) :
            results = list(self.results.values())
            plot_metrics(results_list = results, scenarios_edges = self.scenarios_edges)