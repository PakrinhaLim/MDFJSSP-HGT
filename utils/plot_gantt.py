import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import os

def plot_gantt(cases, action_list, save_path, makespans=None, truncate_breakdowns=False):
    """
    Plots Gantt charts for the scheduling cases.
    action_list should be a list of lists, where each inner list contains:
    [machine, operation, start_time, duration]
    """
    os.makedirs(save_path, exist_ok=True)
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for case_idx, case_actions in enumerate(action_list):
        fig, ax = plt.subplots(figsize=(15, 8))
        case = cases[case_idx]
        
        num_init_ops = sum([job.num_operation for job in case.init_job_list])
        
        y_ticks = []
        y_tick_labels = []
        for i in range(case.num_machine):
            y_ticks.append(i * 10)
            y_tick_labels.append(f"Machine {i}")
            
        for breakdown in case.broken_down_list:
            mac_id, occur_time, duration = breakdown
            
            if makespans is not None and truncate_breakdowns:
                makespan = makespans[case_idx]
                if occur_time >= makespan:
                    continue
                if occur_time + duration > makespan:
                    duration = makespan - occur_time
            
            y = mac_id * 10
            ax.broken_barh([(occur_time, duration)], (y-4, 8), facecolors='none', edgecolors='red', hatch='//', zorder=2)
            
        for action in case_actions:
            if len(action) == 4:
                mac, op, start_time, duration = action
            else:
                continue # Ignore formats that miss duration
                
            if duration > 0:
                y = mac * 10
                # Proxy job classification based on operation index
                job_proxy = op // 5
                c = colors[job_proxy % len(colors)]
                
                if op <= num_init_ops:
                    edge_c = 'black'
                    hatch = None
                else:
                    edge_c = 'blue'
                    hatch = 'xx'
                
                ax.broken_barh([(start_time, duration)], (y-4, 8), facecolors=(c), edgecolors=edge_c, hatch=hatch, zorder=3)
                ax.text(start_time + duration/2, y, f"O_{op}", ha='center', va='center', color='white', fontsize=8, zorder=4)
                
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machines")
        
        makespan_val = f", Makespan: {makespans[case_idx]:.2f}" if makespans is not None else ""
        ax.set_title(f"Gantt Chart - Case {case_idx}{makespan_val}")
        ax.grid(True)
        
        leg_init = mpatches.Patch(facecolor='gray', edgecolor='black', label='Initial Job')
        leg_dyn = mpatches.Patch(facecolor='gray', edgecolor='blue', hatch='xx', label='Dynamic Job')
        leg_brk = mpatches.Patch(facecolor='none', edgecolor='red', hatch='//', label='Machine Breakdown')
        ax.legend(handles=[leg_init, leg_dyn, leg_brk], loc='upper right')
        
        plt.tight_layout()
        chart_file = os.path.join(save_path, f"gantt_case_{case_idx}.png")
        plt.savefig(chart_file, dpi=300)
        plt.close(fig)
        print(f"Saved Gantt chart to {chart_file}")

