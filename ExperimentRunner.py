
from utilities.JobManager import JobManager
from environment.EnvironmentClass import Environment
from environment.MetricsManager import MetricsManager
from utilities.ResultPlotter import plot_metrics
from templates.Scheduler import Scheduler
from RL.agents.Agent import Agent
from RL.CloudEnv import CloudEnv
import copy

class Experiment:
        def __init__(self,
                     infrastructure =  None,
                     jobs = [],
                     scenarios_edges = [],
                     schedulers = [],
                     time_step = 0.01,
                     network_manager = None,
                     network_overhead_enabled = False,
                     power_model = 'DEFAULT'
                ):
            """
            For the power model use these symbols while writing the equations :
            CPU, RAM, STORAGE
            """
            
            self.infrastructure = infrastructure
            self.jobs = jobs
            self.scenarios_edges = scenarios_edges
            self.schedulers = schedulers
            self.time_step = time_step
            self.network_manager = network_manager
            self.power_model = power_model
            
            self.environments = {}
            
            self.results = {}
            
            self.network_overhead_enabled = network_overhead_enabled
            
            last_arrived = max(
                            job.time_arrived for job in self.jobs
                        )
            self.scenarios_edges.append(last_arrived)
            
            if not self.network_overhead_enabled :
                print("\n[INFO] : Network communication overhead is disabled\n"
                      "The experiment is under the assumption of infinite bandwidth\n")
            else :
                print("\n[INFO] : Network communication overhead is enabled\n")        
            
                
        def build_environment(self):
            
            for scheduler in self.schedulers :

                farm = copy.deepcopy(self.infrastructure)
                
                farm.set_power_model(self.power_model)

                job_copy = copy.deepcopy(self.jobs)
                
                network = copy.deepcopy(self.network_manager)
                
                if isinstance(scheduler, Scheduler):
                    self.environments[scheduler] = Environment(
                                                        server_farm=farm,
                                                        metrics_manager=MetricsManager(
                                                            farm,
                                                            jobs=job_copy
                                                        ),
                                                        job_manager=JobManager(
                                                            jobs=job_copy
                                                        ),
                                                        scheduler=scheduler,
                                                        time_step=self.time_step,
                                                        network_manager =  network,
                                                        network_overhead = self.network_overhead_enabled
                                                    )
                if isinstance(scheduler, Agent):
                    self.environments[scheduler] =  CloudEnv(
                                    server_farm=farm,
                                    metrics_manager=MetricsManager(server_farm=farm, jobs=job_copy),
                                    job_manager=JobManager(jobs=job_copy),
                                    agent=scheduler,
                                    network_manager= network,
                                    network_overhead= self.network_overhead_enabled,
                                    time_step= self.time_step
                                )
        
        def run_experiment(self):
            for scheduler in self.schedulers :
                print(f"{scheduler.name} : SCHEDULING - STARTS")
                if isinstance(scheduler, Scheduler):
                    self.results[scheduler] = self.environments[scheduler].run()
                if isinstance(scheduler, Agent):
                    env = self.environments[scheduler]
                    obs, _ = env.reset()
                    done = False

                    while not done:
                        scheduler.observe(obs)
                        action = scheduler.take_action()
                        obs, reward, done, _, _ = env.step(action)
                    self.results[scheduler]  = env.get_results()
                    env.log()
        
        
        def plot_results(self) :
            results = list(self.results.values())
            plot_metrics(results_list = results, scenarios_edges = self.scenarios_edges)