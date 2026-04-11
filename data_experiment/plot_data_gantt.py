import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_data_gantt(data_path, output_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    jobs = data.get('jobs', [])
    
    # We will create a Gantt chart with each Job on a separate row.
    # To show precedence clearly, we'll assume an ASAP schedule where 
    # each task starts exactly when its predecessors finish plus some gap.
    
    fig, ax = plt.subplots(figsize=(16, max(6, len(jobs) * 1.5)))
    
    gap = 2  # Space between operations for arrows
    y_ticks = []
    y_labels = []
    
    colors = plt.cm.tab10.colors
    
    # Track min and max x for limits
    max_endtime = 0
    
    for i, job in enumerate(jobs):
        y_pos = i * 10
        job_id = job['job_id']
        product_type = job.get('product_type', f"Job {job_id}")
        
        y_ticks.append(y_pos + 4)
        # Using a shorter label if possible, or string splitting to make it fit nicely
        y_labels.append(product_type.replace('_', ' '))
        
        operations = job.get('operations', [])
        
        op_times = {} # op_id -> dict
        
        for op in operations:
            op_id = op['operation_id']
            # Using the first available machine's processing time as a proxy for duration
            duration = list(op['processing_times'].values())[0]
            
            preds = op.get('predecessors')
            base_time = 0
            if preds is not None:
                for p in preds:
                    if p in op_times:
                        base_time = max(base_time, op_times[p]['end'])
            
            start_time = base_time + (gap if preds else 0)
            end_time = start_time + duration
            
            op_times[op_id] = {
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'preds': preds
            }
            
            color = colors[op_id % len(colors)]
            
            # Draw Gantt block
            rect = patches.Rectangle((start_time, y_pos), duration, 8, facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # Label
            label_text = f"O_{op_id}\n({duration}m)" if duration >= 25 else f"O_{op_id}"
            ax.text(start_time + duration / 2, y_pos + 4, label_text, 
                    ha='center', va='center', color='black', fontsize=9, fontweight='bold')
            
            # Arrows
            if preds is not None:
                for p in preds:
                    if p in op_times:
                        p_end = op_times[p]['end']
                        # Annotate from predecessor end to this start
                        ax.annotate('', xy=(start_time, y_pos + 4), xytext=(p_end, y_pos + 4),
                                    arrowprops=dict(arrowstyle="->", color='red', lw=1.5))
            
            if end_time > max_endtime:
                max_endtime = end_time

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-5, len(jobs) * 10)
    ax.set_xlim(0, max_endtime + max_endtime * 0.05) # Add 5% padding
    ax.set_xlabel("Time (ASAP Scheduling Approximation, Not Actual Schedule)")
    ax.set_title("Operation Precedence and Relational Gantt Chart for Generated PCB Data")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precedence Gantt chart saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'original_pcb_data.json')
    output_path = os.path.join(base_dir, 'data', 'precedence_gantt.png')
    plot_data_gantt(data_path, output_path)
