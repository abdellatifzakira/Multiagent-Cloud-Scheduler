from collections import deque

def queue_generator(jobs, priority="fifo"):
    """
    Generate an ordered queue of ready tasks from a list of jobs.

    Args:
        jobs     : list of Job objects
        priority : ordering strategy for the queue
                   "fifo"     — insertion order (default)
                   "deadline" — earliest job deadline first
                   "runtime"  — shortest task runtime first (SJF)

    Returns:
        deque of Task objects, ordered by chosen priority
    """
    ready_tasks = []

    for job in jobs:
        tasks = job.get_first_ready_tasks()
        for task in tasks:
            ready_tasks.append((job, task))

    if priority == "deadline":
        ready_tasks.sort(key=lambda x: x[0].deadline)
    elif priority == "runtime":
        ready_tasks.sort(key=lambda x: x[1].runtime)
    # fifo: no sort, insertion order preserved
    return deque(task for job, task in ready_tasks)