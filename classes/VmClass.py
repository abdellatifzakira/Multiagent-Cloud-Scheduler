from collections import deque

class Vm:
    def __init__(self, id: int, c_cpu: float, c_ram: float, compute_power: float):

        self.id = id

        self.cpu = c_cpu
        self.ram = c_ram
        self.compute_power = compute_power


        self.server = None

        self.hosted_task = {}
        self.pending_tasks = deque()

        self.used_cpu = 0.0
        self.used_ram = 0.0
        
        self.max_concurrent_tasks = 100
        self.completed_tasks = 0
        
        self.max_concurrent_tasks_count = 0
        

    # ----------------------------
    # RESOURCE CHECK
    # ----------------------------
    def check_req_constraint(self, task):
        return (
            self.used_cpu + task.cpu <= self.cpu and
            self.used_ram + task.ram <= self.ram and
            len(list(self.hosted_task.values())) + 1 <= self.max_concurrent_tasks
        )
        

        


    # ----------------------------
    # HOST TASK
    # ----------------------------
    def host_task(self, task):     
        
        self.hosted_task[task] = task.job_id

        self.used_cpu += task.cpu
        self.used_ram += task.ram
        
        task.status = 2  # running
        
        return True
    
    
    def time_step_tasks(self,_t, time_step) :
        
        if len(self.hosted_task.keys()) > 0 :
            if len(self.hosted_task.keys()) >= self.max_concurrent_tasks_count :
                self.max_concurrent_tasks_count = len(self.hosted_task.keys())
                
            actual_instruction_rate = time_step*self.compute_power
            #all_cpu_demand = sum(task.cpu for task in self.hosted_task.keys())
            #instruction_per_cpu = actual_instruction_rate/all_cpu_demand
            actual_instruction_rate_per_task = actual_instruction_rate/len(self.hosted_task.keys())
        
        if not self.hosted_task :
            return
        for _task in list(self.hosted_task.keys()):
            #_computed = min(_task.remaining_instructions, instruction_per_cpu*_task.cpu)
            
            _computed = min(_task.remaining_instructions, actual_instruction_rate_per_task)
            _task.remaining_instructions -= _computed
        
        for _task in list(self.hosted_task.keys()):
            _task.advance_timer(_t, time_step)

    # ----------------------------
    # FINISH TASK
    # ----------------------------
    def release_task(self,task, t):
        if task not in list(self.hosted_task.keys()):
            return False

        self.used_cpu -= task.cpu
        self.used_ram -= task.ram
        
        task.on_finished(t) # Inter-tasks notification 
        task.end_time = t
        task.status = 0

        if (self.server is not None) and self.server.network_enabled:
            self.server.save_data(task, t)
        
        self.hosted_task.pop(task)
        if self.server is not None:
            self.server.hosted_tasks.pop(task)
        self.completed_tasks += 1

        return True
    
    
    
    
    
    
    def print_details(self):
        print("CPU : ", self.cpu)
        print("RAM : ", self.ram)
        print("USED CPU : ", self.used_cpu)
        print("USED RAM : ", self.used_ram)


if __name__ == "__main__" :
    from TaskClass import Task
    
    print("="*50)
    print("SINGLE TASK EXECUTION TEST :")
    print("="*50)
    
    task = Task(id=0, job_id=0,
                     cpu = 10, ram = 10,
                     status=1, num_instructions= 1e5,
                     size = 10)
    dt = 0.01
    ndigits = 6
    vm = Vm(
        id = 0,
        c_cpu=20,
        c_ram=20,
        compute_power= 1e6
    )
    t = 0
    if vm.check_req_constraint(task= task):
        vm.host_task(task=task)
        task.vm = vm
        task.start_time = t
        print("TASK HOSTED SUCCESSFULLY")
    print(f"TASK STATUS : {task.status}")
    
    while task.remaining_instructions > 0:
        t += dt
        vm.time_step_tasks(_t=t, time_step=dt)
    print("TASK FINISHED")
    print(f"TASK STATUS : {task.status}")
    print("EXPECTED EXECUTION TIME : 0.1s")
    print("MEASURED EXECUTION TIME : ", round(t, ndigits=ndigits), f"s | ROUNDED TO THE {ndigits}TH DIGIT")  
    print("TASK EXECUTION TIME : ", round(task.end_time - task.start_time, ndigits=ndigits), f"s | ROUNDED TO THE {ndigits}TH DIGIT")  
    
    
    print("="*50)
    print("CONCURRENT TASKS EXECUTION TEST :")  
    print("="*50)
    
    task_1 = Task(id=0, job_id=0,
                     cpu = 10, ram = 10,
                     status=1, num_instructions= 1e5,
                     size = 10)
    
    task_2 = Task(id=0, job_id=0,
                         cpu = 10, ram = 10,
                         status=1, num_instructions= 1e5,
                         size = 10)
    
    vm_concurrent = Vm(
                        id = 0,
                        c_cpu=20,
                        c_ram=20,
                        compute_power= 1e6
                    )
    t = 0
    if vm_concurrent.check_req_constraint(task_1):
        vm_concurrent.host_task(task_1)
        task_1.vm = vm_concurrent
        task_1.start_time = t
        print("HOSTING THE FIRST TASK")
    if vm_concurrent.check_req_constraint(task_2):
            vm_concurrent.host_task(task_2)
            task_2.vm = vm_concurrent
            task_2.start_time = t
            print("HOSTING THE SECOND TASK")
        
    
    
    while vm_concurrent.hosted_task :
        t += dt
        vm_concurrent.time_step_tasks(_t = t, time_step= dt)
        
    
    
    print("EXPECTED EXECUTION TIME : 0.2s")
    print("MEASURED EXECUTION TIME : ", round(t, ndigits=ndigits), f"s | ROUNDED TO THE {ndigits}TH DIGIT")  
    print("EXPECTE EXECUTION PER TASK : 0.2s")
    print("MEASURED EXECUTION TIME PER TASK : ", round(task_1.end_time - task_1.start_time, ndigits=ndigits), f"s | ROUNDED TO THE {ndigits}TH DIGIT") 
    print("MEASURED EXECUTION TIME PER TASK : ", round(task_2.end_time - task_2.start_time, ndigits=ndigits), f"s | ROUNDED TO THE {ndigits}TH DIGIT") 
    
    
    