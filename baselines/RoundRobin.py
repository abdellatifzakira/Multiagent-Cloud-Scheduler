def RoundRobin_schedule(task_queue, vms):
    """
    Schedule tasks to VMs using the Round Robin algorithm.

    Args:
        task_queue (list): A list of tasks to be scheduled.
        vms (list): A list of available VMs.

    Returns:
        dict: A mapping of VM IDs to the tasks assigned to them.
    """
    vm_count = len(vms)
    schedule = {vm.id: [] for vm in vms}  # Initialize schedule dictionary

    for i, task in enumerate(task_queue):
        vm_index = i % vm_count  # Round Robin assignment
        vm_id = vms[vm_index].id
        schedule[vm_id].append(task)  # Assign task to the selected VM

    return schedule