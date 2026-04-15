import random
import numpy as np
from scipy import stats
import json
from utils.case_generator import Operation, sample_truncated_normal, sample_truncated_exponential

class MacroJob:
    def __init__(self, operation_list: list[Operation], total_pieces: int):
        self.operation_list = operation_list
        self.total_pieces = total_pieces
        self.finish_time_min = sum([min(op.operation_times) for op in self.operation_list])
        self.num_operation = len(self.operation_list)
        
    def to_json(self):
        return {
            "total_pieces": self.total_pieces,
            "operations": [op.to_json() for op in self.operation_list]
        }

class MacroCase:
    def __init__(self, init_macro_job_list, suc_macro_job_list, broken_down_list, num_machine):
        self.init_job_list = init_macro_job_list
        self.suc_job_list = suc_macro_job_list
        self.broken_down_list = broken_down_list
        self.num_machine = num_machine
        self.num_operation = sum([job.num_operation for job in self.init_job_list])
        self.num_operation_suc = sum([job.num_operation for job, ar_time in self.suc_job_list])

    def to_json(self):
        return {"init_job_list": [job.to_json() for job in self.init_job_list],
                "suc_job_list": [(job.to_json(), time) for job, time in self.suc_job_list],
                "broken_down_list": self.broken_down_list,
                "num_machine": self.num_machine}

class MacroCaseGenerator:
    def __init__(self,
                 num_machine,
                 num_job,
                 num_operation_max=200,
                 dynamic_prop=0.5,
                 max_break_down=10,
                 lower_num_opj=4,
                 upper_num_opj=7,
                 lower_pres_mu=10,
                 upper_pres_mu=25,
                 sigma_pres_time=1,
                 lower_pres_time=10,
                 upper_pres_time=25,
                 mu_num_cap_mac=1,
                 sigma_num_cap_mac=1,
                 lower_num_cap_mac=1,
                 upper_num_cap_mac=5,
                 mu_lbd=0.01,
                 sigma_lbd=0.005,
                 lower_lbd=0.001,
                 upper_lbd=0.1,
                 lambda_fai_dur=0.1,
                 lower_fai_dur=5,
                 upper_fai_dur=30,
                 lower_fai_itv=50,
                 upper_fai_itv=500,
                 lambda_job_itv=0.1,
                 lower_job_itv=3,
                 upper_job_itv=50):
        self.num_machine = num_machine
        self.num_operation_max = num_operation_max
        self.num_job = num_job

    @staticmethod
    def load_from_pcb_standard_json(json_data, num_cases, args):
        '''
        Parses standard PCB Job-based payload and clusters identical sequences
        into MacroJobs with grouped pieces.
        '''
        cases = []
        for case_idx in range(num_cases):
            macro_jobs_dict = {}
            # deduplicate
            for pcb_job in json_data['jobs']:
                op_sig = []
                operation_list = []
                for op in pcb_job['operations']:
                    sorted_machines = sorted(op['processing_times'].items(), key=lambda x: int(x[0].split('_')[1]))
                    avail_mac = [int(m.split('_')[1]) for m, t in sorted_machines]
                    times = [t for m, t in sorted_machines]
                    op_sig.append(str(avail_mac) + str(times))
                    operation_list.append(Operation(avail_mac, times))
                
                sig = "|".join(op_sig)
                if sig not in macro_jobs_dict:
                    macro_jobs_dict[sig] = {
                        "ops": operation_list,
                        "pieces": 0
                    }
                macro_jobs_dict[sig]["pieces"] += pcb_job.get("pieces_in_batch", 100) # Fallback to 100
                
            init_job_list = []
            for sig, data in macro_jobs_dict.items():
                init_job_list.append(MacroJob(data['ops'], data['pieces']))
            
            # Simple no breakdown structure or dynamic
            case = MacroCase(init_macro_job_list=init_job_list, suc_macro_job_list=[], 
                        broken_down_list=[], num_machine=json_data['nr_machines'])
            cases.append(case)
        return cases
