from generate_pcb_data import PCBDataGenerator
import json
import os
import sys
import argparse
import random

# Add parent dir to sys.path so we can import utils from the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.case_generator import Operation, Job, Case, sample_truncated_exponential, sample_truncated_normal

def load_pcb_as_cases(data, args):
    num_machine = data['nr_machines']
    cases = []
    
    for case_idx in range(args.num_cases):
        all_jobs = []
        
        # Process each job in the PCB dataset to get fresh objects per case
        for pcb_job in data['jobs']:
            operation_list = []
            for op in pcb_job['operations']:
                available_machine = []
                operation_times = []
                
                # Sort by machine id to ensure ordered consistency
                sorted_machines = sorted(op['processing_times'].items(), key=lambda x: int(x[0].split('_')[1]))
                for machine_name, time in sorted_machines:
                    machine_id = int(machine_name.split('_')[1])
                    available_machine.append(machine_id)
                    operation_times.append(time)
                    
                # Create Operation from the extracted mapping
                operation_list.append(Operation(available_machine, operation_times))
            
            # Build Job strictly
            all_jobs.append(Job(operation_list))
            
        # Shuffle jobs for random split into static and dynamic
        random.shuffle(all_jobs)
        
        if args.enable_dynamic:
            num_suc_job = int(len(all_jobs) * args.dynamic_prop)
        else:
            num_suc_job = 0
            
        num_init_job = len(all_jobs) - num_suc_job
        
        init_job_list = all_jobs[:num_init_job]
        suc_job_pool = all_jobs[num_init_job:]
        
        # Dynamically arriving jobs
        suc_job_list = []
        occur_time = 0
        for job in suc_job_pool:
            iterval = sample_truncated_exponential(args.lambda_job_itv, args.lower_job_itv, args.upper_job_itv)
            occur_time += int(iterval.item())
            suc_job_list.append((job, occur_time))
            
        # Machine Breakdown Generation
        broken_down_list = []
        if args.enable_breakdown:
            for mac_id in range(num_machine):
                occur_time = 0
                lbd = sample_truncated_normal(args.mu_lbd, args.sigma_lbd, args.lower_lbd, args.upper_lbd).item()
                for _ in range(args.max_break_down):
                    iterval = sample_truncated_exponential(lbd, args.lower_fai_itv, args.upper_fai_itv).item()
                    occur_time += int(iterval)
                    fai_dur = sample_truncated_exponential(args.lambda_fai_dur, args.lower_fai_dur, args.upper_fai_dur).item()
                    broken_down_list.append([mac_id, occur_time, int(fai_dur)])
                
            # Sort breakdowns chronologically
            broken_down_list.sort(key=lambda x: x[1])
        
        # Wrap all jobs in a single Case
        case = Case(init_job_list=init_job_list, suc_job_list=suc_job_list, 
                    broken_down_list=broken_down_list, num_machine=num_machine)
        cases.append(case)
        
    return cases

def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate comprehensive dataset from PCB data")
    parser.add_argument('--num_cases', type=int, default=10, help='Number of cases to generate')
    parser.add_argument('--enable_dynamic', type=bool, default=False, help='Enable dynamic job arrivals')
    parser.add_argument('--dynamic_prop', type=float, default=0.2, help='Proportion of dynamically arriving jobs')
    parser.add_argument('--enable_breakdown', type=bool, default=False, help='Enable machine breakdowns')
    parser.add_argument('--max_break_down', type=int, default=1, help='Max number of breakdowns per machine')
    
    # Breakdown distribution parameters
    parser.add_argument('--mu_lbd', type=float, default=0.001)
    parser.add_argument('--sigma_lbd', type=float, default=0.0005)
    parser.add_argument('--lower_lbd', type=float, default=0.0001)
    parser.add_argument('--upper_lbd', type=float, default=0.005)
    
    # Breakdown duration parameters
    parser.add_argument('--lambda_fai_dur', type=float, default=0.1)
    parser.add_argument('--lower_fai_dur', type=float, default=5)
    parser.add_argument('--upper_fai_dur', type=float, default=30)
    
    # Breakdown interval parameters
    parser.add_argument('--lower_fai_itv', type=float, default=2000)
    parser.add_argument('--upper_fai_itv', type=float, default=5000)
    
    # Dynamic job arrival parameters
    parser.add_argument('--lambda_job_itv', type=float, default=0.1)
    parser.add_argument('--lower_job_itv', type=float, default=3)
    parser.add_argument('--upper_job_itv', type=float, default=50)
    
    args = parser.parse_args()
    
    pcb_data = PCBDataGenerator()
    pcb_data.processing_info
    pcb_data.save_dataset(os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'original_pcb_data.json')))
    # Destination output location tailored for eval.py/run.py compatibility
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'multilayer_pcb_dataset_standard.json'))
    
    print(f"Generating {args.num_cases} cases with {int(args.dynamic_prop * 100)}% dynamic jobs.")
    
    cases = load_pcb_as_cases(pcb_data.processing_info, args)
    
    # Convert Case objects back to strictly typed JSON format used by MDFJSSP-HGT
    cases_json = [case.to_json() for case in cases]
    
    with open(output_file, 'w') as f:
        json.dump(cases_json, f, indent=4)
        
    print(f"✅ Successfully exported normalized data.")
    print(f"💾 Saved standard MDFJSSP instance to: {output_file}")
    print("\n🚀 You can now evaluate this instance utilizing the following command:")
    print("python eval.py --data_name multilayer_pcb_dataset_standard")
