import igraph as ig

class Job:
    def __init__(
        self,
        id: int,
        dag: ig.Graph,
        tasks: list,
        num_tasks: int,
        time_arrived: float,
    ):
        self.id = id
        self.dag = dag
        self.tasks = self.populate_tasks(tasks)
        self.num_tasks = num_tasks
        self.time_arrived = time_arrived
        self.time_completed = None
        self.get_first_ready_task_flag = False
        self.deadline = self.find_critical_path_length()

    # ─────────────────────────────────────────────
    # PROPERTIES
    # ─────────────────────────────────────────────
    @property
    def completed(self):
        return all(task.status == 0 for task in self.tasks.values())

    @property
    def number_of_rejected_tasks(self):
        return sum(1 for task in self.tasks.values() if task.status == -1)

    # ─────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────
    def populate_tasks(self, tasks):
        # FIX: always key by int(task.id) — original mixed int and str keys
        # depending on which method was called, causing silent KeyErrors
        return {task.id: task for task in tasks}

    # ─────────────────────────────────────────────
    # READY TASK LOGIC
    # ─────────────────────────────────────────────
    def get_ready_tasks(self):
        if not self.get_first_ready_task_flag:
            self.get_first_ready_task_flag = True
            return self.get_first_ready_tasks()

        ready_tasks = set()
        for task_id, task in self.tasks.items():
            if task.status == 0:
                children = self.get_children_of_task(task_id)
                ready_tasks.update(children)
        return list(ready_tasks)

    def get_first_ready_tasks(self):
        """Return tasks with no incoming edges (DAG entry points)."""
        ready = []
        for task_id, task in self.tasks.items():
            # FIX: use int task_id directly — original used str() here
            if len(self.dag.neighbors(task_id, mode="in")) == 0:
                task.status = 1
                ready.append(task)
        return ready

    def get_children_of_task(self, task_id):
        """
        Return child tasks of task_id that are not yet completed,
        and mark them as ready (status=1).

        FIX: original used setattr() as a side-effect inside a list
        comprehension condition — this is a well-known Python anti-pattern
        that works but is fragile and unreadable. Replaced with a plain loop.

        FIX: all dict lookups now use int keys consistently.
        """
        children = []
        for child_id in self.dag.neighbors(task_id, mode="out"):
            child_id = int(child_id)  # igraph may return different int types
            if child_id not in self.tasks:
                continue
            child_task = self.tasks[child_id]
            if child_task.status == 0:
                continue  # already completed, skip
            child_task.status = 1  # unlock: mark as ready
            children.append(child_task)
        return children

    def get_parent_of_task(self, task_id):
        """Return parent tasks of task_id."""
        parents = []
        for parent_id in self.dag.neighbors(task_id, mode="in"):
            parent_id = int(parent_id)
            if parent_id in self.tasks:
                parents.append(self.tasks[parent_id])
        return parents

    # ─────────────────────────────────────────────
    # REJECTION
    # ─────────────────────────────────────────────
    def reject_task_and_cascade(self, task_id):
        """Reject a task and cascade rejection to all downstream dependents."""
        if task_id not in self.tasks or self.tasks[task_id].status == -1:
            return

        stack = [task_id]
        visited = set()

        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            self.tasks[current_id].status = -1
            stack.extend(self.get_future_dependent_tasks(current_id))

    def get_future_dependent_tasks(self, task_id):
        """Return downstream task IDs that are still pending (status 1 or 3)."""
        future = []
        for neighbor_id in self.dag.neighbors(task_id, mode="out"):
            neighbor_id = int(neighbor_id)
            if neighbor_id not in self.tasks:
                continue
            if self.tasks[neighbor_id].status in (1, 3):
                future.append(neighbor_id)
                future.extend(self.get_future_dependent_tasks(neighbor_id))
        return future

    # ─────────────────────────────────────────────
    # CRITICAL PATH
    # ─────────────────────────────────────────────
    def find_critical_path_length(self):
        """
        Longest path through the DAG by total runtime (critical path length).
        Uses dynamic programming on the topological sort.

        FIX: original indexed max_weight by raw vertex ID assuming 0-based
        contiguous integers — now uses a dict so it works regardless of
        how igraph numbers vertices.

        FIX: all task lookups use int keys.
        """
        topological_order = self.dag.topological_sorting()

        # dp[vertex_id] = longest cumulative runtime to reach this vertex
        dp = {v: 0.0 for v in topological_order}

        # seed with source vertices (no incoming edges)
        for v in topological_order:
            if len(self.dag.neighbors(v, mode="in")) == 0:
                dp[v] = self.tasks[int(v)].runtime

        for v in topological_order:
            for neighbor in self.dag.neighbors(v, mode="out"):
                neighbor = int(neighbor)
                candidate = dp[v] + self.tasks[neighbor].runtime
                if candidate > dp[neighbor]:
                    dp[neighbor] = candidate

        return max(dp.values()) if dp else 0.0

    def modify_task_status(self, task_id, status):
        # FIX: use int key directly
        self.tasks[int(task_id)].status = status


def print_dag(job):
        print("\nDAG Structure:")
        for task_id in job.tasks:
            children = job.dag.neighbors(task_id, mode="out")
            children = [int(c) for c in children]
            if children:
                for child in children:
                    print(f"  Task {task_id} → Task {child}")
            else:
                print(f"  Task {task_id} → (no children / leaf node)")


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from task import Task
    dag = ig.Graph(directed=True)
    dag.add_vertices(5)
    dag.add_edges([(0, 1), (2, 1), (1, 3), (2, 3), (3, 4)])

    tasks = [
        Task(id=0, job_id=1, cpu=2.0, ram=4.0, status=3, runtime=5.0),  # initialized
        Task(id=1, job_id=1, cpu=1.0, ram=2.0, status=3, runtime=3.0),  # initialized
        Task(id=2, job_id=1, cpu=1.5, ram=3.0, status=3, runtime=4.0),  # initialized
        Task(id=3, job_id=1, cpu=2.5, ram=5.0, status=3, runtime=6.0),  # initialized
        Task(id=4, job_id=1, cpu=3.0, ram=6.0, status=3, runtime=7.0),  # initialized
    ]

    job = Job(id=1, dag=dag, tasks=tasks, num_tasks=len(tasks), time_arrived=10.0)
    
    """
    The job has the following DAG structure:
      
        0   2
         \ /
          1
          |
          3
          |
          4
    """
    deadline = job.find_critical_path_length()
    print(f"Critical path length (deadline): {deadline}")  # Expected: 21
    first_ready_tasks = job.get_first_ready_tasks()
    print(f"First ready tasks: {[task.id for task in first_ready_tasks]}")  # Expected: [0, 2]
    #Get the DAG structure of the job
    print_dag(job)