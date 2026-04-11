# MDFJSSP-HGT Feature Dimensions and Architecture Deep Dive

## Constructed Feature Dimensions

Based on the project's source code (`env/mdfjssp.py` and `env/base_class.py`), here is the exact breakdown of the feature dimensions constructed for states:

### 1. `operation_raw_feature` (7 Dimensions)
This feature vector captures the real-time and static statuses of every operation.
*   **Dimension 0 (`[..., 0]`) - Operation Status**: 
    *   `0`: The operation is available or waiting.
    *   `1`: The operation is currently processing.
*   **Dimension 1 (`[..., 1]`) - Average Processing Time**: The average processing time for this operation calculated across all the machines it can be assigned to.
*   **Dimension 2 (`[..., 2]`) - Progress Ratio in Job**: The relative position of the operation within its job, calculated as `(operation_index_in_job + 1) / total_operations_in_job`. 
*   **Dimension 3 (`[..., 3]`) - Processed Time Left**: The remaining processing time for the operation if it is currently being processed (`action_timer`). It is `0` if not yet started or already finished.
*   **Dimension 4 (`[..., 4]`) - Earliest Estimated Completion Time**: The earliest estimated time the entire job can finish (specifically, the estimated completion time of the last operation within the same job).
*   **Dimension 5 (`[..., 5]`) - Available Machines Count**: The total number of alternative machines that this operation can be processed on.
*   **Dimension 6 (`[..., 6]`) - Remaining Operations in Job**: The number of subsequent operations left in the job after this current one.

### 2. `machine_raw_feature` (4 Dimensions)
This feature vector captures the state of every machine in the environment.
*   **Dimension 0 (`[..., 0]`) - Machine Status**:
    *   `0`: The machine is free/available.
    *   `1`: The machine is occupied (currently processing an operation).
    *   `-1`: The machine is currently broken down.
*   **Dimension 1 (`[..., 1]`) - Connectable Operations**: The total number of unfinished and available operations left in the environment that can potentially be processed on this machine.
*   **Dimension 2 (`[..., 2]`) - Elapsed Processing Time**: The amount of time that has already passed for the operation currently assigned to this machine (`operation_time - action_timer`).
*   **Dimension 3 (`[..., 3]`) - Break Down Frequency**: A ratio representing the breakdown history of the machine over time (`machine_broken_record / current_time`).

### 3. `arc_raw_feature` (2 Dimensions)
This feature maps the 2D relationship/connection between machines and operations.
*   **Dimension 0 (`[..., 0]`) - Specific Assignment Processing Time**: The exact time an operation takes on a specific machine. Missing/Invalid arcs are filled with the maximum processing time limit to deter those connections.
*   **Dimension 1 (`[..., 1]`) - Active Elapsed Assign Time**: The elapsed execution time, specifically tracking connections that are currently actively processing (`(operation_time - action_timer) * sign(action_timer)`).

---

## Heterogeneous Graph Attention Network (HGAT) Architecture

The raw features act as the foundational building blocks for the **Heterogeneous Graph Attention Network (HGAT)** (`policy/hgan.py`), which acts as the core mathematical brain behind the PPO agent. 

### 1. Scaling & Embedding Layer (`normalize_raw_feature`)
Because the raw dimensions vary wildly in scale, injecting them directly into an attention mechanism would cause destabilization.

*   **Batch Normalization & Sigmoid Check:** The 7 Operation dims, 4 Machine dims, and 2 Arc dims are run through separate `BatchNorm1d` layers and compressed into a `0` to `1` scalar ratio using a Sigmoid function (`w_o`, `w_m`, `w_arc`).
*   **Weighted Embeddings:** These normalized features act as "gates" or "weights" that dynamically scale fixed, learnable embedding weights (`operation_embedding`, `machine_embedding`, `arc_embedding`), projecting local rules into a rich continuous space (`d_model`).

### 2. The Interaction Encoders (Node <-> Arc Fusion)
Before message-passing in the graph, the network explicitly fuses nodes and arcs:
*   `h_arc_o`: The operation embeddings (`h_o`) are mathematically concatenated with the arc embeddings (`h_arc`) through a Linear fusion layer to represent the overall "Operation-side graph view."
*   `h_arc_m`: The machine embeddings (`h_m`) are concatenated with the arc embeddings to represent the overall "Machine-side graph view."

### 3. The Heterogeneous Attention Mechanism
The core of the HGAT iterates `N` times (defined by `num_layers`), exchanging information across the bipartite graph:
*   **Machine Encoder Sub-network (`HGTMachineEncoderLayer`)**: 
    Machine nodes update their internal state using Multi-Head Attention. The *Query* is the Machine state (`h_m`), while the *Keys/Values* are the fused operation-arcs (`h_arc_o`). In simpler terms: **"Machines look at all valid Operations they can process, weighing what those operations require, and update their own scheduling bottleneck awareness."**
*   **Operation Encoder Sub-network (`HGTOperationEncoderLayer`)**:
    Operation nodes act as the *Query* (`h_o`), pulling information from the fused machine-arcs (`h_arc_m`). Furthermore, this is where **Job Precedence** is solved:
    ```python
    h_o_next = adj_matrix.float().bmm(self.w_next(self.norm4(h_o)))
    ```
    The operations pass their features forward mathematically along the `adj_matrix` (which tracks operation sequences in jobs). This ensures operation $k$ restricts or informs operation $k+1$.

### 4. Splitting into Actor & Critic (PPO Iterations)
From these deep graph structures, the network branches into the reinforcement learning components:

*   **The Actor Sub-network (Action Prediction / `get_prop`)**: 
    The network calculates a massive attention score matrix between **every final machine feature** against **every final operation feature**. It masks out impossible or finished assignments, and runs a standard `Softmax` over it mapping a perfect probability map of which Machine should process which Operation next.
*   **The Critic Sub-network (`evaluate` value prediction)**: 
    It globally pools unfinished operations and machine features to construct `h_global`, an isolated snapshot of the overall environment state. It pushes `h_global` through an MLP (`evaluate_mlp`) to predict the expected total completion time (the 'value' for PPO advantage calculation).
