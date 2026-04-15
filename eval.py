import argparse
import json
import os

from utils.evaluate import eval
from utils.utils import set_device, set_seed


def main(args_parsed):
    # set seed
    set_seed(args_parsed['seed'])
    # set device
    device = set_device(args_parsed['gpu'])
    makespan, act_list, makespans = eval(args_parsed, device)
    file_name = args_parsed['prefix'] if args_parsed['save_name'] is None else args_parsed['save_name']
    
    if args_parsed['visualize']:
        out_dir = os.path.dirname(args_parsed['save_gantt_path'])
        if not out_dir:
            out_dir = '.'
        os.makedirs(out_dir, exist_ok=True)
        json.dump(act_list, open(args_parsed['save_gantt_path'] + file_name + 'actions.json', 'w'))

        # Load cases explicitly to pass into plot_gantt
        from utils.case_generator import CaseGenerator
        from utils.plot_gantt import plot_gantt
        
        cases_test = json.load(open(f"./data/{args_parsed['data_name']}.json", "r"))
        cases_test = CaseGenerator.from_json(cases_test)
        
        gantt_save_path = os.path.join(out_dir, "gantt_charts", args_parsed['data_name'], args_parsed['checkpoint_id'])
        plot_gantt(cases_test, act_list, gantt_save_path, makespans=makespans, truncate_breakdowns=args_parsed.get('truncate_breakdowns', False))

        scheduling_data = {
            'makespan': float(makespan) if makespan is not None else None,
            'makespans': [float(m) for m in makespans] if makespans is not None else None,
            'act_list': act_list
        }
        with open(os.path.join(gantt_save_path, f"{file_name}_scheduling_data.json"), 'w') as f:
            json.dump(scheduling_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic Flexible Job Scheduling')
    # main
    parser.add_argument('--data', type=str, default="valid")
    parser.add_argument('--data_name', type=str, default="valid")
    parser.add_argument('--save_path', type=str, default="./checkpoint/")
    parser.add_argument('--prefix', type=str, default="default")
    parser.add_argument('--policy', type=str, default='hgt')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--checkpoint_id', type=str, default='default')
    # environment
    parser.add_argument('--num_machine', type=int, default=10)
    parser.add_argument('--num_job', type=int, default=20)
    parser.add_argument('--num_operation_max', type=int, default=200)
    parser.add_argument('--num_cases', type=int, default=20)
    parser.add_argument('--dynamic_prop', type=float, default=0.3)
    parser.add_argument('--sigma_pres_time', type=float, default=1)
    parser.add_argument('--mu_lbd', type=float, default=0.01)
    # policy

    parser.add_argument('--use_macro_env', action='store_true', help='Use the Option 2 continuous macro flow graph')
    parser.add_argument('--d_operation_raw', type=int, default=9)
    parser.add_argument('--d_machine_raw', type=int, default=4)
    parser.add_argument('--d_arc_raw', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_hidden', type=int, default=256)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--d_kv', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.4)
    # ppo
    parser.add_argument('--gamma_m', type=float, default=0.98)
    parser.add_argument('--gamma_u', type=float, default=0.925)
    parser.add_argument('--lambda', type=float, default=1.0)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--policy_loss_coefficient', type=float, default=1)
    parser.add_argument('--value_function_loss_coefficient', type=float, default=1)
    parser.add_argument('--entropy_loss_coefficient', type=float, default=0.01)
    # train args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--case_regen_iter', type=int, default=5)
    parser.add_argument('--epochs_per_iter', type=int, default=3)
    parser.add_argument('--early_stop', type=int, default=None)
    parser.add_argument('--warmup_step', type=int, default=None)
    # general setting
    parser.add_argument("--gpu", type=int, default=-2,
                        help="Use the GPU with the lowest memory footprint by default. "
                             "Specify a GPU by setting this parameter to a GPU id which equal to or greater than 0."
                             "Set this parameter to _1 to use the CPU."
                        )
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed.")
    # visualize
    parser.add_argument("--visualize", type=bool, default=True,
                        help="visualize the gantt chart.")
    parser.add_argument("--save_gantt_path", type=str, default="saved_plots/",
                        help="save path for gantt chart.")
    parser.add_argument("--truncate_breakdowns", type=bool, default=False,
                        help="Whether to cut off machine breakdowns at the makespan time.")
    args_parsed = parser.parse_args()
    args_parsed = vars(args_parsed)
    main(args_parsed)
