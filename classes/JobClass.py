import numpy as np
import igraph as ig
try :
    from classes.TaskClass import Task
except ModuleNotFoundError: 
    from TaskClass import Task

class Job:
    _counter = 0
    def __init__(
        self,
        tasks: list  = None,
        time_arrived: float = None,
        data_transfer_weights: dict = None,
        id: int = None,
        sla_factor= 1.5,

    ):
        if id is None:
            self.id = Job._counter
            Job._counter += 1
        else:
            self.id = id
            Job._counter = max(Job._counter, id + 1)
        self.tasks = self.populate_tasks(tasks)
        self.time_arrived = time_arrived
        self.data_transfer_weights = data_transfer_weights
        self.total_tasks = len(self.tasks) if tasks is not None else 0
        self.dag = self.build_dag()
        self.end_time = None
        self.sla_violated = None
        self.counted = False
        self.success = False
        self.sla_factor = sla_factor
        self.sla_limit = None
    def set_time_arrived(self, _t):
        self.time_arrived = _t
        
        for _tsk in self.tasks.values():
            if _tsk.remaining_parents == 0:
                _tsk.arrival_time = _t
        
        
    
    def build_dag(self):
        if self.data_transfer_weights is None:
            return None
        edges = list(self.data_transfer_weights.keys())
        weights = list(self.data_transfer_weights.values())
        vertices = set()
        
        tasks = self.tasks.values()
        
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)

        assert len(vertices) <= self.total_tasks, "tasks weights mismatch!"

        graph = ig.Graph(directed=True)

        # assume task ids are integers starting from 0
        graph.add_vertices(self.total_tasks)
        graph.add_edges(edges)

        # assign weights
        graph.es["weight"] = weights
        
        # assign attributes
        num_instructions = [task.num_instructions for task in tasks]
        cpus = [task.cpu for task in tasks]
        rams = [task.ram for task in tasks]
        graph.vs["num_instructions"] = num_instructions
        graph.vs["cpu"] = cpus
        graph.vs["ram"] = rams
        

        return graph

    def spawn_task(self, task):
        task.job_id = self.id
        self.tasks[task.id] = task
        
    def populate_tasks(self, tasks):
        if tasks is None :
            return None
        for task in tasks:
            task.job_id = self.id
        return {task.id: task for task in tasks}
    

    @staticmethod
    def spawn_job(num_tasks=3,
                  job_id = None,
                  cpu_req = [32, 32, 32],
                  ram_req = [10, 10, 10],
                  instructions = [5e6, 5e6, 5e6],
                  data_transfer_weights = None,
                  arrival_time = None,
                  sizes = [32,32,32]
                  ):
        assert num_tasks == len(sizes) == len(cpu_req) == len(ram_req) == len(instructions), f"\n[INPUT ERROR] Input length mismatch \nEXPECTED TASKS NUM = {num_tasks} \nCPU REQ LENGTH = {len(cpu_req)} \nRAM REQ LENGTH = {len(ram_req)} \nINSTRUCTIONS NUM LENGTH = {len(instructions)}\nTASKS SIZES LENGTH = {len(sizes)}"
        
        Tasks = [ Task(
            id = i,
            job_id=job_id,
            cpu = cpu_req[i],
            ram= ram_req[i],
            status= 3,
            num_instructions = instructions[i],
            size= sizes[i]
                )
            for i in range(num_tasks)]
        
        
        job = Job(
            id = job_id,
            tasks=Tasks,
            data_transfer_weights=data_transfer_weights,
            time_arrived=arrival_time
        )
        
        
        # assign children and parents to task :
        if data_transfer_weights is not None :
            for parent_key, child_key in data_transfer_weights.keys():
                job.tasks[parent_key].children.append(job.tasks[child_key])
                job.tasks[child_key].parents.append(job.tasks[parent_key])
                
                job.tasks[child_key].parent_weights[job.tasks[parent_key]] = data_transfer_weights[(parent_key, child_key)]
            
        for tsk in job.tasks.values() :
            tsk.remaining_parents = len(tsk.parents)
            if tsk.remaining_parents == 0 :
                tsk.arrival_time = job.time_arrived
                tsk.status = 1
        
        job.sla_limit = (
                            job.get_critical_path_runtime()
                            * job.sla_factor
                        )
        
        for task in Tasks :
            task.job_sla =  job.sla_limit
            task.job_arrival = job.time_arrived
        return job
    
    
    
    @staticmethod
    def generate_jobs(num_jobs, num_tasks_per_job=4,
                      time_arrived = [], edge_probability = 0.5,
                      instructions_per_task = [250e6, 1e9],
                      data_transfer_range = [512, 1024]):
        """
        Generate multiple jobs with random parameters.
        
        Args:
            num_jobs: number of jobs to create
            num_tasks_per_job: tasks per job (default 4)
        
        Returns:
            list of Job objects
        """
        
        if time_arrived is not None :
            assert len(time_arrived) == num_jobs, \
                "[INVALID INPUT] : Arrival times does not macth the jobs number."
        
        jobs = []
        for job_id in range(num_jobs):
            
            # Random task parameters
            cpu_req = [round(np.random.uniform(12, 64), 0) for _ in range(num_tasks_per_job)]
            ram_req = [round(np.random.uniform(2, 32), 0) for _ in range(num_tasks_per_job)]
            instructions = [round(np.random.uniform(min(instructions_per_task),max(instructions_per_task)), 0) for _ in range(num_tasks_per_job)]
            sizes = [round(np.random.uniform(32, 128), 0) for _ in range(num_tasks_per_job)]
            
            min_data = min(data_transfer_range)
            max_data = max(data_transfer_range)
            # Generate random DAG edges
            data_transfer_weights = {}
            for i in range(num_tasks_per_job - 1):
                for j in range(i + 1, num_tasks_per_job):
                    if np.random.random() < edge_probability:  # chance of edge
                        weight = np.random.randint(min_data,max_data)
                        data_transfer_weights[(i, j)] = weight
            
            job = Job().spawn_job(
                num_tasks=num_tasks_per_job,
                cpu_req=cpu_req,
                ram_req=ram_req,
                instructions=instructions,
                sizes= sizes,
                data_transfer_weights=data_transfer_weights if data_transfer_weights else None,
                job_id=job_id,
                arrival_time=time_arrived[job_id]
            )
            
            

            jobs.append(job)
        
        return jobs



    
    def get_total_cpu_req(self):
        return round(sum(task.cpu for task in self.tasks.values()), ndigits=2)
    
    def get_total_ram_req(self):
        return round(sum(task.ram for task in self.tasks.values()),ndigits=2)
    
    def get_total_req(self):
        return self.get_total_cpu_req(), self.get_total_ram_req()
    
    def build_tree(self, graph, node):
        children = graph.neighbors(node, mode="out")

        if not children:
            return {} 

        return {
            child: self.build_tree(graph, child)
            for child in children
        }
    
    def describe_job(self):
        graph = self.dag
        origins = [v.index for v in graph.vs if graph.degree(v, mode="in") == 0]
        trees = {}
        for origin in origins:
            trees[origin] = self.build_tree(graph, origin)
        return trees
    
    
    def print_job_layout(self, tree, graph, level=0):
        indent = "  " * level
        if tree is None:
            return
        for node, subtree in tree.items():
            if subtree != {}:
                print(f"{indent}{node}")
                for child in subtree:
                    # get edge ID
                    eid = graph.get_eid(node, child)
                    weight = graph.es[eid]["weight"]
                    print(f"{indent} =======({weight})=====> {child}")
                self.print_job_layout(subtree, graph, level + 1)
    
    
    def get_direct_paths_cost(self):
        graph = self.dag
        roots = [v.index for v in graph.vs if graph.degree(v, mode="in") == 0]
        all_paths = {}
        def dfs(node, path, cost):
            children = graph.neighbors(node, mode="out")
            # if leaf → store path
            if len(children) == 0:
                all_paths[tuple(path)] = cost
                return
            for child in children:
                eid = graph.get_eid(node, child)
                weight = graph.es[eid]["weight"]
                dfs(child, path + [child], cost + weight)
        for r in roots:
            dfs(r, [r], 0)
        return all_paths
    
    
    
    def get_direct_paths_cpu_req(self):
        graph = self.dag
        roots = [v.index for v in graph.vs if graph.degree(v, mode="in") == 0]

        all_paths = {}

        def dfs(node, path, cpu):
            children = graph.neighbors(node, mode="out")

            if len(children) == 0:
                all_paths[tuple(path)] = cpu
                return

            for child in children:
                child_cpu = graph.vs[child]["cpu"]
                dfs(child, path + [child], round(cpu + child_cpu, ndigits=2))

        for r in roots:
            root_cpu = graph.vs[r]["cpu"]
            dfs(r, [r], root_cpu)

        return all_paths
    
    
    def get_direct_paths_ram_req(self):
        graph = self.dag
        roots = [v.index for v in graph.vs if graph.degree(v, mode="in") == 0]

        all_paths = {}

        def dfs(node, path, ram):
            children = graph.neighbors(node, mode="out")

            if len(children) == 0:
                all_paths[tuple(path)] = ram
                return

            for child in children:
                child_ram = graph.vs[child]["ram"]
                dfs(child, path + [child], round(ram + child_ram,ndigits=2))

        for r in roots:
            root_ram = graph.vs[r]["ram"]
            dfs(r, [r], root_ram)

        return all_paths
    
    def get_costly_path(self):
        costs = self.get_direct_paths_cost()
        if not costs:
            return None
        most_costly_path = max(costs.items(), key=lambda item: item[1])
        return most_costly_path
    
    
    
    def get_cpu_demanding_path(self):
        cpus = self.get_direct_paths_cpu_req()
        if not cpus:
            return None
        max_cpu = max(cpus.items(), key=lambda item: item[1])
        return max_cpu
    
    def get_ram_demanding_path(self):
        rams = self.get_direct_paths_ram_req()
        if not rams:
            return None
        max_ram = max(rams.items(), key=lambda item: item[1])
        return max_ram 

    def get_entry_points(self):
        entries = []
        for task in self.tasks.values():
            if len(task.parents) == 0:
                entries.append(task)
        return entries
    
    def get_ready_tasks(self):
        ready = []
        for task in self.tasks.values():
            if task.status == 1:
                ready.append(task)

        return ready

    def get_running_tasks(self):
        running = []
        for task in self.tasks.values():
            if task.status == 2:
                running.append(task)
        

        return running
    
    
    
    def get_critical_path_runtime(self):
        
        _compute_power_refrence = 25e6 # 25_000_000 instruction/second a baseline : modest vm

        if self.dag is None:
            return max(
                (task.num_instructions/_compute_power_refrence)
                for task in self.tasks.values()
            )

        earliest_finish = {}

        # topological order
        topo = self.dag.topological_sorting()

        for node in topo:

            task_instructions = self.tasks[node].num_instructions

            parents = self.dag.predecessors(node)

            if len(parents) == 0:
                earliest_start = 0

            else:
                earliest_start = max(
                    earliest_finish[p]
                    for p in parents
                )

            earliest_finish[node] = earliest_start + task_instructions


        return max(earliest_finish.values())/_compute_power_refrence

