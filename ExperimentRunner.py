
from utilities.JobManager import JobManager
from environment.EnvironmentClass import Environment
from environment.MetricsManager import MetricsManager
from utilities.ResultPlotter import plot_metrics
from templates.Scheduler import Scheduler
from RL.agents.Agent import Agent
from RL.CloudEnv import CloudEnv
import copy
import numpy as np

class Experiment:
        def __init__(self,
                     infrastructure =  None,
                     jobs = [],
                     scenarios_edges = [],
                     schedulers = [],
                     time_step = 0.01,
                     network_manager = None,
                     network_overhead_enabled = False,
                     power_model = 'DEFAULT',
                     evaluation = None,
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
            
            self.evaluation = evaluation
            
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
                                    time_step= self.time_step,
                                    evaluation = self.evaluation
                                )
        
        def run_experiment(self):
            for scheduler in self.schedulers :
                print(f"{scheduler.name} : SCHEDULING - STARTS")
                if isinstance(scheduler, Scheduler):
                    self.results[scheduler] = self.environments[scheduler].run()
                if isinstance(scheduler, Agent):
                    env = self.environments[scheduler]
                    episodes = 0
                    if scheduler.trainable :
                        episodes = 5
                    for episode in range(episodes+1):
                        if episode == episodes:
                            env.mod = 'TEST'
                        env.reward_buffer = []
                        state, _ = env.reset(episode = episode)
                        done = False
                        while not done:
                            #if env.is_scheduling_time():
                                state, _ = env.get_state()
                                scheduler.observe(
                                    state
                                )
                                action = scheduler.take_action()
                                next_state, reward, done = env.step(action)
                                if scheduler.trainable :
                                    next_state = np.array(next_state, dtype=np.float32)
                                    scheduler.update(
                                        action,
                                        reward,
                                        next_state,
                                        done
                                    )
                            #else:
                            #    env.step_ahead()
                        if scheduler.trainable:
                            print(f"{env.mod = }, {episode = }, MEAN CPU STD : {-round(float(np.mean(env.reward_buffer)), ndigits= 5)}")
                            scheduler.epsilon = max(0.05, scheduler.epsilon*0.95)
                    self.results[scheduler]  = env.get_results()
                    env.log()
        
        
        def plot_results(self) :
            results = list(self.results.values())
            plot_metrics(results_list = results, scenarios_edges = self.scenarios_edges)