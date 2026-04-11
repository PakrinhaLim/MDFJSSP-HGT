import argparse
import torch
import json
import os
from env.mdfjssp import MDFJSSPEnv
from utils.plot_graph import plot_state_graph
from utils.case_generator import CaseGenerator

def main():
    args = {
        'num_machine': 40, # As per valid_d04s1f001_m5j15
        'num_job': 1000,
        'num_operation_max': 200,
        'num_cases': 100,
        'dynamic_prop': 0.3, # Ignored, data is predefined
        'sigma_pres_time': 1,
        'mu_lbd': 0.01,
        'd_operation_raw': 7,
        'd_machine_raw': 4,
        'd_arc_raw': 2,
    }

    # Load predefined test Case
    data_name = "multilayer_pcb_dataset_standard"
    with open(f"./data/{data_name}.json", "r") as f:
        cases_json = json.load(f)
    print(f"Loaded cases from {data_name}.json")

    cases = CaseGenerator.from_json(cases_json)
    
    # We will just plot the first case from the batch
    env = MDFJSSPEnv(args, cases, device='cuda')
    state = env.reset()
    
    # Process the initial state to load in the first jobs
    env.process()
    env.handle_new_job()
    env.handle_break_down()
    
    # Create saved_plots directory if it doesn't exist
    os.makedirs('saved_plots', exist_ok=True)
    
    plot_state_graph(state, batch_idx=0, save_path=f'saved_plots/example_graph_{data_name}.png')

if __name__ == "__main__":
    main()
