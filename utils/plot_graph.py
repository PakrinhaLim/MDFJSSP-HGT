import networkx as nx
import matplotlib.pyplot as plt
import torch

def plot_state_graph(state, batch_idx=0, save_path='graph.png'):
    """
    Plots the graph representation of the State object for a given batch index
    using a layered neural-network-like layout.
    """
    mask_machine = state.mask_machine[batch_idx].squeeze(-1).cpu().numpy()
    mask_operation = state.mask_operation[batch_idx].squeeze().cpu().numpy()
    mask_machine_operation = state.mask_machine_operation[batch_idx].cpu().numpy()
    operation_adj_matrix = state.operation_adj_matrix[batch_idx].cpu().numpy()
    
    machine_features = state.machine_raw_feature[batch_idx].cpu().numpy()
    operation_features = state.operation_raw_feature[batch_idx].cpu().numpy()

    num_machine = len(mask_machine)
    num_operation_max = len(mask_operation)

    G = nx.DiGraph()
    pos = {}

    # 1. Pre-calculate operation depths (layers)
    op_G = nx.DiGraph()
    for j in range(1, num_operation_max):
        if mask_operation[j]:
            op_G.add_node(j)
            
    for i in range(1, num_operation_max):
        if not mask_operation[i]: continue
        for j in range(1, num_operation_max):
            if not mask_operation[j]: continue
            if operation_adj_matrix[i, j]:
                op_G.add_edge(i, j)

    op_labels = {}
    job_idx = 1
    start_nodes = sorted([n for n in op_G.nodes() if op_G.in_degree(n) == 0])
    for start in start_nodes:
        curr = start
        op_idx = 1
        while curr is not None:
            op_labels[curr] = f"O{job_idx},{op_idx}"
            successors = list(op_G.successors(curr))
            curr = successors[0] if successors else None
            op_idx += 1
        job_idx += 1

    # Calculate depth using longest path from nodes with 0 in-degree
    depths = {node: 1 for node in op_G.nodes()}
    try:
        for node in nx.topological_sort(op_G):
            for neighbor in op_G.successors(node):
                if depths[neighbor] < depths[node] + 1:
                    depths[neighbor] = depths[node] + 1
    except nx.NetworkXUnfeasible:
        pass # In case of cycle
    
    max_depth = max(depths.values()) if depths else 1

    # 2. Add Start and End Nodes
    G.add_node("Start", type='special', label="Start")
    G.add_node("End", type='special', label="End")
    pos["Start"] = (0, 0)
    pos["End"] = (max_depth + 1, 0)

    # 3. Position Operations by depth
    ops_by_depth = {}
    for node, d in depths.items():
        if d not in ops_by_depth:
            ops_by_depth[d] = []
        ops_by_depth[d].append(node)

    y_gap = 2
    for d, ops in ops_by_depth.items():
        num_ops = len(ops)
        start_y = (num_ops - 1) * y_gap / 2.0
        for idx, j in enumerate(ops):
            node_id = f"O{j}"
            label = op_labels.get(j, node_id)
            G.add_node(node_id, label=label, type='operation')
            pos[node_id] = (d, start_y - idx * y_gap)
            
            # Connect to Start
            if op_G.in_degree(j) == 0:
                G.add_edge("Start", node_id, edge_type='precedence', color='black')
            # Connect to End
            if op_G.out_degree(j) == 0:
                G.add_edge(node_id, "End", edge_type='precedence', color='black')

    # Add precedence edges
    for u, v in op_G.edges():
        G.add_edge(f"O{u}", f"O{v}", edge_type='precedence', color='blue')

    # 4. Position Machines at the top
    active_machines = [i for i in range(num_machine) if mask_machine[i]]
    num_active_machines = len(active_machines)
    
    machine_y = max([p[1] for p in pos.values()]) + 4 if pos else 10
    machine_x_start = 1
    machine_x_gap = max_depth / max(1, num_active_machines - 1) if num_active_machines > 1 else max_depth / 2.0

    for idx, i in enumerate(active_machines):
        node_id = f"M{i}"
        status = round(machine_features[i, 0], 2)
        label = f"{node_id}\nStat:{status}"
        G.add_node(node_id, label=label, type='machine')
        pos[node_id] = (machine_x_start + idx * machine_x_gap, machine_y)

    # Add machine-operation edges
    for i in active_machines:
        for j in op_G.nodes():
            if mask_machine_operation[i, j]:
                G.add_edge(f"M{i}", f"O{j}", edge_type='machine_operation', color='gray')

    # 5. Draw the graph
    plt.figure(figsize=(16, 10))
    
    color_map = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'operation')
        if node_type == 'machine':
            color_map.append('lightblue')
        elif node_type == 'special':
            color_map.append('lightgray')
        else:
            color_map.append('lightgreen')

    labels = {node: G.nodes[node]['label'] for node in G.nodes()}

    edges_mo = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'machine_operation']
    edges_prec = [(u, v) for u, v, d in G.edges(data=True) if d.get('edge_type') == 'precedence']

    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2000, edgecolors='black')
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    # Draw machine-operation capability lines (dashed, gray)
    nx.draw_networkx_edges(G, pos, edgelist=edges_mo, edge_color='gray', style='dashed', alpha=0.3, arrows=False)
    # Draw precedence lines (solid, blue)
    nx.draw_networkx_edges(G, pos, edgelist=edges_prec, edge_color='blue', style='solid', arrows=True, connectionstyle='arc3,rad=0.1')

    plt.title(f"MDFJSSP State Graph Diagram - Layered Layout")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    print(f"Graph visualization saved to {save_path}")
