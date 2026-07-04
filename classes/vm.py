class Vm:
  def __init__(
    self,
    id: int,
    cpu: float,
    ram: float
  ):
    
    self.id = id
    
    self.cpu = cpu
    
    self.ram = ram
    
    self.status = 0 # 0: off, 1: occupied
    
    self.hosted_task = None
    
    self.used_cpu = 0.0
    self.used_ram = 0.0
  
  def host_task(self, task):
    self.hosted_task = task
    self.status = 1  # Set VM status to 1: occupied
    self.cpu += task.cpu
    self.ram += task.ram
    self.used_cpu += task.cpu
    self.used_ram += task.ram

  def finish_task(self):
    assert self.hosted_task is not None, "VM has not hosted task, expect a hosted task"
    if self.status == 1 and self.hosted_task:
      self.used_cpu -= self.hosted_task.cpu
      self.used_ram -= self.hosted_task.ram
      self.status = 0
      hosted_task= self.hosted_task
      hosted_task.status = 0  # Mark the task as finished
      vm_id = self.id
      self.hosted_task = None
      
    return hosted_task