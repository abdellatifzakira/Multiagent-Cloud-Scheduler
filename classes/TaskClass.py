class Task:
  def __init__(
    self,
    id: int,
    job_id: int,
    cpu: float,
    ram: float,
    status: int,
    runtime: float,
    size: int
    ):
    
    self.id = id
    self.job_id = job_id
    self.cpu = cpu
    self.ram = ram
    self.server_farm_id = None
    self.server_id = None
    self.vm_id = None
    self.status = status # -1: rejected, 0: finished, 1: ready, 2: running, 3: initialized.
    self.runtime = runtime
    self.arrival_time = None
    self.start_time = None
    self.end_time = None
    self.parents = []
    self.children =  []
    self.remaining_parents = 0
    self.monitored = False
    self.meet_sla = None
    self.size = None
    
    
    
    
  def notify_parent_finished(self, _t):
    if self.remaining_parents > 0:
        self.remaining_parents -= 1

    if self.remaining_parents == 0:
        self.status = 1  # READY
        self.arrival_time = _t
  
  def on_finished(self, _t):
    for child in self.children:
        child.notify_parent_finished(_t)
      
    #self.deadline = self.start_time + self.runtime