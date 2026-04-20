# Paper Titles

Here are a few academic titles ranging from descriptive to punchy:

1. **Learning to Dispatch in Multilayer PCB Manufacturing via Heterogeneous Graph Attention Networks and Proximal Policy Optimization** *(Very descriptive, good for IEEE/CIRP)*
2. **A Heterogeneous Graph Reinforcement Learning Approach for Large-Scale Flexible Job Shop Scheduling with Batch Processing** *(Focuses on the algorithm and the batching aspect)*
3. **Scalable Neural Combinatorial Optimization for Massive PCB Fabrication Facilities using HGAT-PPO** *(Strong keywords, good for Operations Research journals)*
4. **Graph-Based Deep Reinforcement Learning for Dynamic Rescheduling in High-Mix Multilayer PCB Manufacturing** *(Focuses on the application and dynamic nature)*

---

# Paper Outline & Draft

## Abstract
Modern multilayer Printed Circuit Board (PCB) manufacturing involves complex Flexible Job Shop Scheduling Problems (FJSP) characterized by parallel machines, sequence-dependent flows, and batch-processing machines (e.g., lamination presses). Traditional exact solvers fail to scale to industrial dataset sizes (e.g., thousands of operations), while heuristic dispatching rules often yield suboptimal makespans. In this paper, we propose a novel deep reinforcement learning (DRL) framework using a Heterogeneous Graph Attention Network (HGAT) combined with Proximal Policy Optimization (PPO). By representing the shop floor as a heterogeneous graph—distinguishing between machine nodes and operation queues—our model efficiently captures complex spatial and temporal constraints. Experimental results on large-scale PCB datasets demonstrate that our HGAT-PPO approach significantly outperforms traditional heuristics and scales effectively to massive instances where exact solvers (CP-SAT) time out.

## 1. Introduction
* **Context:** The growing complexity of multilayer PCBs (HDI, RF boards) and the need for high-mix, low-volume manufacturing.
* **The Problem:** Scheduling PCB fabrication is uniquely difficult because it combines continuous flow (plating lines), discrete parallel routing (CNC drilling), and batch-processing (lamination presses). 
* **The Gap:** Exact solvers (MIP, CP) are strictly limited by instance size. Metaheuristics (GA, TS) are too slow for real-time dynamic rescheduling. Standard Graph Neural Networks (GNNs) suffer from over-smoothing and memory issues on massive graphs.
* **Contribution:** We formulate the large-scale PCB scheduling problem as a Markov Decision Process (MDP) and solve it using an HGAT-based PPO agent, utilizing sub-batching and hetero-graph representations to achieve state-of-the-art inference speeds.

## 2. Problem Formulation
* **FJSP-B (Flexible Job Shop with Batching):** Define the mathematical model. $N$ jobs, $M$ machines.
* **Multilayer PCB Specifics:** Detail the specific routing of multilayer boards (DES $\rightarrow$ AOI $\rightarrow$ Press $\rightarrow$ Drill $\rightarrow$ Plating $\rightarrow$ Mask $\rightarrow$ Router $\rightarrow$ Test).
* **Objective Function:** Minimize the maximum completion time (Makespan - $C_{max}$).

## 3. Methodology
### 3.1 Heterogeneous Graph Representation
Unlike standard homogenous representations, we construct a Heterogeneous Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
* **Node Types:** Operation Nodes $\mathcal{V}_O$ (representing sub-batched operations) and Machine Nodes $\mathcal{V}_M$.
* **Edge Types:** 
  * `Precedence Edges` ($\mathcal{V}_O \rightarrow \mathcal{V}_O$): Defining flow.
  * `Eligibility/Allocation Edges` ($\mathcal{V}_O \leftrightarrow \mathcal{V}_M$): Connecting operations to eligible parallel machines.

### 3.2 Heterogeneous Graph Attention Network (HGAT)
* Formulate the message-passing equations. Show how attention weights $\alpha_{ij}$ are calculated differently depending on whether the edge connects two operations or an operation to a machine. This allows the network to learn bottleneck congestion and starvation independently.

### 3.3 Proximal Policy Optimization (PPO) Setup
* **State:** The extracted HGAT node embeddings.
* **Action:** When a machine $m$ becomes idle, the agent selects an eligible operation $o$ from the connected Operation nodes.
* **Reward Structure:** Dense reward formulation (e.g., negative increment of machine idle times) combined with sparse terminal makespan rewards.

## 4. Experiments and Results
* **Dataset Generation:** Sub-batched datasets modeling 5,000+ pieces across 40 PCB machines.
* **Baselines:** Compare the agent against FIFO (First In First Out), SPT (Shortest Processing Time), a Genetic Algorithm (GA), and Google OR-Tools (CP-SAT).
* **Metrics:** Makespan, Inference Time (seconds to solve), and Generalization (training on 50 jobs, testing on 100 jobs).

---

# Coding Architecture 

To implement this paper, your codebase should be split into 3 tightly coupled modules.

## 1. The RL Environment (`pcb_env.py`)
Use OpenAI `Gymnasium`. This script steps through time.
```python
import gymnasium as gym

class PCBSchedulingEnv(gym.Env):
    def __init__(self, jobShopEnv):
        self.jobShopEnv = jobShopEnv
        # Track time, machine availability, and completion status
        self.current_time = 0
        
    def step(self, action):
        # Action is: (machine_id, operation_id)
        # 1. Update the operation's completion time
        # 2. Fast-forward time to the next machine event
        # 3. Calculate Reward (idle time penalty)
        # 4. Check if all jobs are done (terminated)
        return next_state_graph, reward, terminated, truncated, info
```

## 2. The HGAT Model (`hgat_model.py`)
Using `PyTorch Geometric` (`torch_geometric`), define the Actor-Critic networks.
```python
import torch
from torch_geometric.nn import HGTConv, Linear

class HGATActorCritic(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()
        # Heterogeneous Graph Transformer (HGT) allows complex msg passing
        self.convs = torch.nn.ModuleList()
        # Define metadata (node types and edge types)
        metadata = (['operation', 'machine'], 
                    [('operation', 'precedes', 'operation'), 
                     ('operation', 'eligible', 'machine')])
                     
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum'))
            
        # Actor Head (Outputs probabilities for each operation)
        self.actor_head = Linear(hidden_channels, 1)
        # Critic Head (Evaluates the state for the Advantage function)
        self.critic_head = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # x_dict contains node features for "operations" and "machines"
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            
        # The actor looks at the Operation nodes to select which one goes next
        action_logits = self.actor_head(x_dict['operation'])
        state_value = self.critic_head(x_dict['machine'].mean(dim=0)) # Pool machine states
        
        return action_logits, state_value
```

## 3. The PPO Training Loop (`train_ppo.py`)
The orchestrator that collects rollouts and updates the network.
```python
import torch.optim as optim
from torch.distributions import Categorical

def train_ppo(env, model, epochs):
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    for epoch in range(epochs):
        # 1. Collect trajectories (states, actions, log_probs, rewards)
        states, actions, log_probs, rewards, values = collect_rollouts(env, model)
        
        # 2. Calculate Advantages (Generalized Advantage Estimation)
        advantages = calculate_gae(rewards, values)
        
        # 3. PPO Clipping objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        critic_loss = torch.nn.MSELoss()(values, returns)
        
        # 4. Backpropagation
        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
