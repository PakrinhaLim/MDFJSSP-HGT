import torch
import gymnasium as gym
from env.base_class import State, Action
from utils.macro_case_generator import MacroCase

class MDFJSSPMacroEnv(gym.Env):
    def __init__(self, args, cases: list[MacroCase], device='gpu'):
        self.num_machine = args['num_machine']
        self.num_operation_max = args['num_operation_max']
        self.num_cases = args['num_cases']
        self.d_operation_raw = args['d_operation_raw']  
        self.d_machine_raw = args['d_machine_raw']
        self.d_arc_raw = args['d_arc_raw']
        self.device = device
        self.cases = cases
        
        self.cur_time = 0
        self.state = None
        
    def reset(self, keep_cases=True, seed=None) -> State:
        self.cur_time = 0
        self.num_operation_max = max([case.num_operation for case in self.cases]) + 1
        
        self.num_machine = self.cases[0].num_machine
        # Initialize Tensors
        self.queue_sizes = torch.zeros((self.num_cases, self.num_operation_max), dtype=torch.long, device=self.device)
        self.remaining_pieces = torch.zeros((self.num_cases, self.num_operation_max), dtype=torch.long, device=self.device)
        self.next_op_map = torch.ones((self.num_cases, self.num_operation_max), dtype=torch.long, device=self.device) * -1
        
        mask_machine = torch.ones((self.num_cases, self.num_machine, 1), dtype=torch.bool, device=self.device)
        mask_broken_down = torch.zeros((self.num_cases, self.num_machine, 1), dtype=torch.bool, device=self.device)

        mask_operation = torch.zeros((self.num_cases, 1, self.num_operation_max), dtype=torch.bool, device=self.device)
        mask_operation_finished = torch.zeros((self.num_cases, 1, self.num_operation_max), dtype=torch.bool, device=self.device)
        mask_operation_finished[:, :, 0] = True # Op 0 is padding generally
        
        mask_operation_processing = torch.zeros((self.num_cases, 1, self.num_operation_max), dtype=torch.bool, device=self.device)
        mask_machine_operation = torch.zeros((self.num_cases, self.num_machine, self.num_operation_max), dtype=torch.bool, device=self.device)
        
        operation_raw_feature = torch.zeros(self.num_cases, self.num_operation_max, self.d_operation_raw, device=self.device)
        machine_raw_feature = torch.zeros(self.num_cases, self.num_machine, self.d_machine_raw, device=self.device)
        arc_raw_feature = torch.zeros(self.num_cases, self.num_machine, self.num_operation_max, self.d_arc_raw, device=self.device)
        
        accu_matrix = torch.zeros((self.num_cases, self.num_operation_max, self.num_operation_max), dtype=torch.long, device=self.device)
        operation_adj_matrix = torch.zeros((self.num_cases, self.num_operation_max, self.num_operation_max), dtype=torch.bool, device=self.device)
        operation_time = torch.zeros((self.num_cases, self.num_machine, self.num_operation_max), dtype=torch.long, device=self.device)
        last_operation_per_job = torch.zeros((self.num_cases, 1, self.num_operation_max), dtype=torch.long, device=self.device)
        
        for i in range(self.num_cases):
            operation_bias = 1
            for j, macro_job in enumerate(self.cases[i].init_job_list):
                op_count = macro_job.num_operation
                last_operation_per_job[i, :, operation_bias:operation_bias + op_count] = operation_bias + op_count - 1
                
                # First operation gets the full queue of initial pieces
                self.queue_sizes[i, operation_bias] = macro_job.total_pieces
                
                for k, operation in enumerate(macro_job.operation_list):
                    op_idx = operation_bias + k
                    self.remaining_pieces[i, op_idx] = macro_job.total_pieces
                    
                    if k < op_count - 1:
                        self.next_op_map[i, op_idx] = op_idx + 1
                        
                    operation_raw_feature[i, op_idx, 2] = (k + 1) / op_count
                    operation_raw_feature[i, op_idx, 5] = len(operation.available_machine)
                    operation_raw_feature[i, op_idx, 6] = op_count - k - 1
                    
                    mask_machine_operation[i, operation.available_machine, op_idx] = True
                    operation_time[i, operation.available_machine, op_idx] = torch.LongTensor(operation.operation_times).to(self.device)
                    mask_operation[i, 0, op_idx] = True
                    operation_adj_matrix[i, 0 if k == 0 else op_idx - 1, op_idx] = True
                    accu_matrix[i, operation_bias:op_idx + 1, op_idx] = 1
                    
                operation_bias += op_count

        temp = operation_time.clone()
        temp[temp == 0] = torch.iinfo(operation_time.dtype).max
        temp = (temp.min(dim=1)[0]).unsqueeze(1)
        operation_finished_time_earliest = (temp.float() @ accu_matrix.float()).long()
        
        self.state = State(operation_raw_feature=operation_raw_feature, machine_raw_feature=machine_raw_feature,
                      arc_raw_feature=arc_raw_feature, mask_broken_down=mask_broken_down,
                      mask_machine=mask_machine, mask_machine_operation=mask_machine_operation,
                      mask_operation=mask_operation, mask_operation_processing=mask_operation_processing,
                      mask_operation_finished=mask_operation_finished, operation_adj_matrix=operation_adj_matrix,
                      operation_time=operation_time, accu_matrix=accu_matrix,
                      machine_broken_record=torch.zeros(self.num_cases, self.num_machine, 1, device=self.device),
                      operation_finished_time_earliest=operation_finished_time_earliest,
                      last_operation_per_job=last_operation_per_job)

        self.action_timer = torch.zeros((self.num_cases, self.num_machine, self.num_operation_max), dtype=torch.long, device=self.device)
        self.machine_broken_timer = torch.zeros((self.num_cases, self.num_machine, 1), dtype=torch.long, device=self.device)
        
        self.machine_running_op = torch.ones((self.num_cases, self.num_machine), dtype=torch.long, device=self.device) * -1
        
        self.state.update_feature(self.action_timer, self.cur_time)
        return self.state

    def process(self):
        # find timers that just hit 0 from > 0
        finished_machines = (self.action_timer_prev > 0) & (self.action_timer == 0)
        
        for c in range(self.num_cases):
            for m in range(self.num_machine):
                if finished_machines[c, m].any().item():
                    op = self.machine_running_op[c, m].item()
                    if op != -1:
                        # piece finished processing
                        self.remaining_pieces[c, op] -= 1
                        nxt = self.next_op_map[c, op].item()
                        if nxt != -1:
                            self.queue_sizes[c, nxt] += 1
                        self.machine_running_op[c, m] = -1
        
        # update state masks dynamically for macro flow
        # operation is finished when no pieces remain to process
        mask_op_fin = (self.remaining_pieces == 0).unsqueeze(1)
        mask_op_fin[:, 0, 0] = True # Keep padding 0 true
        
        # operation is processing if any machine is engaged on it
        mach_engaged = self.action_timer > 0
        mask_op_proc = mach_engaged.any(dim=1, keepdim=True)
        
        mask_mach = ~mach_engaged.any(dim=2, keepdim=True)
        
        self.state.update(mask_operation_processing=mask_op_proc, mask_machine=mask_mach, mask_operation_finished=mask_op_fin)
        
        # Update custom queues
        self.state.operation_raw_feature[:, :, 7] = self.queue_sizes.float()
        self.state.operation_raw_feature[:, :, 8] = self.remaining_pieces.float()

    def get_runnable_cases_macro(self):
        # Action requires a free machine, AND an operation with queue > 0 AND machine available
        has_queue = self.queue_sizes > 0
        # mask available machines
        mach_free = self.state.mask_machine.squeeze(2) # (cases, mach)
        
        # shape computation for available (cases, mach, op)
        # can this machine do this op?
        can_do = self.state.mask_machine_operation & mach_free.unsqueeze(2) & has_queue.unsqueeze(1)
        runnable = can_do.any(dim=-1).any(dim=-1)
        return can_do, runnable

    def step(self, actions: Action):
        # execute assignment
        self.action_timer_prev = self.action_timer.clone()
        runnable = actions.runnable_cases
        acts = actions.actions[runnable]
        mach_idx = acts[:, 0]
        op_idx = acts[:, 1]
        
        # Apply assignment
        case_idx = torch.where(runnable)[0]
        self.action_timer[case_idx, mach_idx, op_idx] = self.state.operation_time[case_idx, mach_idx, op_idx]
        self.machine_running_op[case_idx, mach_idx] = op_idx
        self.queue_sizes[case_idx, op_idx] -= 1
        
        self.action_timer_prev = self.action_timer.clone() # sync before loop

        # Loop time
        can_do, run_cases = self.get_runnable_cases_macro()
        finished = (self.remaining_pieces[:, 1:] == 0).all(dim=1).all().item()
        
        while (not run_cases.any().item()) and (not finished):
            self.cur_time += 1
            self.action_timer_prev = self.action_timer.clone()
            self.action_timer = torch.clamp(torch.sub(self.action_timer, 1), min=0)
            self.process()
            
            can_do, run_cases = self.get_runnable_cases_macro()
            finished = (self.remaining_pieces[:, 1:] == 0).all(dim=1).all().item()
            
        self.state.update_feature(self.action_timer, self.cur_time, run_cases)
        # Mock zero reward for simplification unless requested 
        reward = torch.zeros(self.num_cases, 2, device=self.device)
        return self.state, reward, finished, finished, None
