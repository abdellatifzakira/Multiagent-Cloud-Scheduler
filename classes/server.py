import math

from numpy import empty

class Server:
  def __init__(
    self,
    id: int,
    server_farm_id: int,
    vms: list,
    c_cpu: float,
    c_ram: float,
    alpha: float,
    beta: float,
    static_power: float = 0.035,
    optimal_utilization_rate: float = 0.7
  ):
    
    self.id = id
    
    self.server_farm_id = server_farm_id
    
    self.vms = self.populate_vm(vms)
    
    self.vm_numbers = len(self.vms)
    
    self.c_cpu = c_cpu
    
    self.c_ram = c_ram
    
    self.alpha = alpha
    
    self.beta = beta

    self.static_power_value = static_power

    self.optimal_utilization_rate = optimal_utilization_rate
    
    self.current_cpu_usage = 0.0
    
    self.current_ram_usage = 0.0
    
  # Energy Consumption Model on Server
  # Inspired from:
  # "DRL-Cloud: Deep Reinforcement Learning-Based Resource Provisioning
  # and Task Scheduling for Cloud Service Providers".
  @property
  def cpu_utilization_rate(self):
    total_vms_cpu = sum(vm.used_cpu for vm in self.vms.values() if vm.status == 1) 
    cpu_utilization_rate = total_vms_cpu / self.c_cpu 
    return round(cpu_utilization_rate, 2)

  @property
  def static_power(self):
    return self.static_power_value if self.cpu_utilization_rate > 0 else 0

  @property
  def dynamic_power(self):
    cpu_utilization_rate = self.cpu_utilization_rate
    optimal_utilization_rate = self.optimal_utilization_rate
    if cpu_utilization_rate < optimal_utilization_rate:
      return round(cpu_utilization_rate * self.alpha, 2)
    return round(
      (optimal_utilization_rate * self.alpha) +
      (math.pow(cpu_utilization_rate - optimal_utilization_rate, 2)
      * self.beta), 2)

  @property
  def total_power(self):
    return round((self.static_power + self.dynamic_power), 2)
  
  @property
  def is_available(self):
    # Check if any VM in the server is available
    return any(vm.status == 0 for vm in self.vms.values())
  
  def populate_vm(self, vms):
    return {vm.id: vm for vm in vms}
  
  def host_task_in_server(self, task):
    available_vm_id = [vm.id for vm in self.vms.values() if vm.status == 0]
    
    if available_vm_id: # take the available vm id
      for vm_id in available_vm_id:
        verdict = self.check_cpu_mem_constraint(task, vm_id) # check task CPU and MEM constraint
        if verdict: # if task can be hosted without violating VM and MEM limits, host the task on VM
          self.vms[vm_id].host_task(task)
          task.vm_id = vm_id
          task.server_id = self.id
          task.server_farm_id = self.server_farm_id
          self.vms[vm_id].status = 1  # Mark the VM as occupied
          
          # perform check if the task actually exists in the hosted VM
          #print("vm id that host the task: ", self.vms[vm_id].id)
          
          
          self.current_cpu_usage += self.vms[vm_id].cpu
          self.current_ram_usage += self.vms[vm_id].ram
          return True, task # True for successful task hosting
    return False # False for unsuccessful task hosting
  # check if the VM has sufficient CPU or RAM to host the task without overburdening VM resource constraint
  def check_cpu_mem_constraint(self, task, vm_id):
    vm = self.vms[vm_id]
    if (task.cpu + vm.used_cpu > vm.cpu) or (task.ram + vm.used_ram > vm.ram):
      return False
    return True
  
  def clear_completed_task_in_server(self, task):
    vm = self.vms[task.vm_id]
    task_id, vm_id = vm.release_task()
    
    self.current_cpu_usage -= self.vms[vm_id].cpu
    self.current_ram_usage -= self.vms[vm_id].ram
    
    return task_id, vm_id
  
  def check_active_vms(self):
    for vm in self.vms.values():
      if vm.status == 0:
        print(f"VM {vm.id} : is active on server {self.id}")
      else :
        print(f"VM {vm.id} : is inactive on server {self.id}")
  def activate_vms(self):
    for vm in self.vms.values():
      vm.status = 0
  
  def deactivate_vms(self):
    for vm in self.vms.values():
      vm.status = 1
  
  def toggle_vm_status(self, vm_id, state=0):
    if vm_id in self.vms:
      self.vms[vm_id].status = state

# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from vm import Vm
    from task import Task

    # Create VMs for the server
    vms = [Vm(id=1, cpu=0.5, ram=0.5), Vm(id=2, cpu=0.3, ram=0.3)]

    # Create a server
    server = Server(id=1, server_farm_id=1, vms=vms, c_cpu=1.0, c_ram=1.0, alpha=0.5, beta=0.3)

    # Create tasks
    task1 = Task(id=1, job_id=1, cpu=0.4, ram=0.4, status=3, runtime=5.0)
    task2 = Task(id=2, job_id=1, cpu=0.6, ram=0.6, status=3, runtime=3.0)

    # Host tasks in the server
    result1 = server.host_task_in_server(task1)
    print(f"Task 1 hosting result: {result1}")

    result2 = server.host_task_in_server(task2)
    print(f"Task 2 hosting result: {result2}")

    # Check active VMs
    server.check_active_vms()

    # Clear completed task
    completed_task_id, vm_id = server.clear_completed_task_in_server(task1)
    print(f"Completed Task ID: {completed_task_id}, VM ID: {vm_id}")

    # Check active VMs after clearing task
    server.check_active_vms()