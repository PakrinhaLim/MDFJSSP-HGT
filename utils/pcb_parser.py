import json
import os
import sys

def parse_pcb_dataset(input_path, output_path):
    print(f"Loading PCB dataset from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)
        
    num_machine = data.get("nr_machines", 40)
    jobs = data.get("jobs", [])
    
    init_job_list = []
    
    for job in jobs:
        # Sort operations by predecessor sequence/operation_id to ensure order
        # Assuming the order is correctly represented by operation_id
        operations = job.get("operations", [])
        operations = sorted(operations, key=lambda x: x["operation_id"])
        
        job_ops_list = []
        for op in operations:
            available_machine = []
            operation_times = []
            
            for machine_key, time_val in op.get("processing_times", {}).items():
                mac_id = int(machine_key.split("_")[1])
                available_machine.append(mac_id)
                operation_times.append(time_val)
                
            # They need to be sorted by machine index
            combined = sorted(zip(available_machine, operation_times), key=lambda x: x[0])
            sorted_available_machine = [x[0] for x in combined]
            sorted_operation_times = [x[1] for x in combined]
            
            job_ops_list.append({
                "available_machine": sorted_available_machine,
                "operation_times": sorted_operation_times
            })
            
        init_job_list.append(job_ops_list)
        
    # The JSSP code expects a list of cases where each case is a dictionary
    case = {
        "num_machine": num_machine,
        "init_job_list": init_job_list,
        "suc_job_list": [],
        "broken_down_list": []
    }
    
    cases = [case]
    
    print(f"Successfully parsed 1 case with {num_machine} machines and {len(init_job_list)} jobs.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cases, f, indent=4)
        
    print(f"Saved parsed dataset to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data_experiment", "data", "multilayer_pcb_dataset.json")
    output_file = os.path.join(base_dir, "data", "multilayer_pcb_parsed.json")
    
    parse_pcb_dataset(input_file, output_file)
