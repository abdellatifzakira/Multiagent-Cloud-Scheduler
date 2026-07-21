import random
class Task:
  def __init__(
    self,
    id: int,
    job_id: int,
    cpu: float,
    ram: float,
    status: int,
    num_instructions: int,
    size: int = None, 
    ):
    
    self.id = id
    self.job_id = job_id
    self.cpu = cpu
    self.ram = ram
    self.server_farm_id = None
    self.server = None
    self.status = status # -1: rejected, 0: finished, 1: ready, 2: running, 3: initialized, 4: pending.
    self.num_instructions = num_instructions
    self.arrival_time = None
    self.start_time = None
    self.end_time = None
    self.parents = []
    self.children =  []
    self.remaining_parents = 0
    self.monitored = False
    self.meet_sla = None
    self.size = size
    self.completion_time = 0
    self.vm = None
    self.left_retry = 3
    self.remaining_instructions = num_instructions
    self.parent_weights = {}
    self.job_sla = None
    self.job_arrival = None
    self.data_requested = False
    self.scheduled = False
    self.time_data_arrival = None
    
  
  def advance_timer(self, _t, time_step) :
    if self.status != 2 :
      return
    if self.remaining_instructions > 0 :
      self.completion_time += time_step
    else :
      self.vm.release_task(self, _t)
    
    
  def notify_parent_finished(self, _t):
    if self.remaining_parents > 0:
        self.remaining_parents -= 1

    if self.remaining_parents <= 0:
        self.status = 1  # READY
        self.arrival_time = _t
  
  def on_finished(self, _t):
    for child in self.children:
        child.notify_parent_finished(_t)
      
    
  
  
  def fail_and_cascade(self) :
    if self.vm is not None:
      self.vm.release_failed_task(self)
    if self.children :
      for _child in self.children :
        _child.fail_and_cascade()
        
  
  def reset(self) :
    self.server_farm_id = None
    self.server = None
    self.remaining_parents = len(self.parents)
    self.status = 1 if len(self.parents) <= 0 else 3
    if len(self.parents) > 0 :
      self.arrival_time = None 
    self.start_time = None
    self.end_time = None
    self.monitored = False
    self.meet_sla = None
    self.timer = None
    self.vm = None
    self.left_retry = 3