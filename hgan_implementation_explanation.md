# MDFJSSP `hgan.py` Implementation Overview

The `policy/hgan.py` file implements the actor-critic network for the PPO agent using a **Heterogeneous Graph Attention Network (HGAN)**. In this scheduling problem, the environment is modeled as a heterogeneous graph where **Operations** and **Machines** are the nodes, and the processing relations (which operation can run on which machine, and how long it takes) are the edges/arcs. The goal of this network is to extract features from this graph and decide the best operation-machine pairing.

## 1. Fundamental Building Blocks

- **`RMSNorm`**: A custom implementation of Root Mean Square Normalization layer, which is a variant of LayerNorm that is often faster and more stable in Transformer architectures.
- **`MultiHeadAttention`**: A multi-head self-attention mechanism, similar to the one used in standard Transformers. However, it takes `h` (node features) as queries and `h_arc` (edge/arc features) as keys and values. This allows nodes to gather information based on the specific relation they have with other nodes.
- **`FFN`**: A standard two-layer Feed-Forward Network with a GELU activation function and dropout.

## 2. Graph Encoder Layers

These classes define the heterogeneous message passing mechanisms between Machines and Operations.

- **`HGTMachineEncoderLayer`**: Computes attention to pass messages from the "Operation/Arc" embeddings to the "Machine" embeddings. The machine learns its current status based on the operations available to it.
- **`HGTOperationEncoderLayer`**: Defines how an operation node gathers messages. However, there is a special addition here: `h_o_next = adj_matrix...`. This uses the `adj_matrix` (which encodes the sequence/order of operations in a job) to pass information from the *next* operation backward, ensuring the current operation knows if a critical step is waiting for it to finish.

## 3. The Main Network: `HGAN`

This is the core neural network model that glues everything together to output probabilities and values.

### Initialization & Embeddings (`__init__` and `normalize_raw_feature`)

- It takes the raw, normalized features for Machines (`d_machine_raw`), Operations (`d_operation_raw`), and Arcs (`d_arc_raw`).
- These features are pushed through `nn.Embedding` (or standard Linear layers if they are pre-normalized continuous values) to map them into a shared latent dimension (`d_model`).

### The Encoder (`encode`)

```python
def encode(self, state: State):
    h_o, h_m, h_arc = self.normalize_raw_feature(state)
    ...
```

This function is the core graph convolution block. It loops for `num_layers`:
1. It fuses the current Operation features (`h_o`) with the Arc features (`h_arc`) into `h_arc_o`.
2. It fuses the current Machine features (`h_m`) with the Arc features (`h_arc`) into `h_arc_m`.
3. The Machine nodes update their representations by attending to the Operations (`machine_encoder_layers`).
4. The Operation nodes update their representations by attending to the Machines and their successor Operations (`operation_encoder_layers`).

### The Actor (Policy / Action Selection) (`get_prop` & `forward`)

```python
def get_prop(self, state: State): ...
def forward(self, state: State, sample=True): ...
```

- After encoding, the network applies highly parameterized linear layers (`w_m`, `w_o`) to the final learned Machine and Operation representations.
- It performs a batch matrix multiplication (`h_m_.bmm(h_o_)`) between the Operation vectors and Machine vectors. 
- The resulting matrix essentially contains a "score" for every possible Operation-Machine pairing.
- An `action_mask` is applied to set invalid assignments (like assigning an operation to a machine that cannot process it) to negative infinity (`-inf`).
- Finally, it applies `Softmax` over these scores to get the action probabilities. `Categorical(prop).sample()` selects an action based on these probabilities during training.

### The Critic (Value Estimation) (`evaluate`)

```python
def evaluate(self, state: State, actions: torch.Tensor):
    ...
    values = self.evaluate_mlp(h_global)
```

- In PPO reinforcement learning, an agent needs a "critic" to estimate how good its current state is (the Value function).
- This function calculates a `h_global` representation by mean-pooling the features of all unfinished Operations (`pooled_ho`) and all Machines (`pooled_hm`).
- It concatenates these global representations and passes them through a Multi-Layer Perceptron (the `evaluate_mlp` defined in `__init__`) to output a single scalar: the expected future return (Value) from this state.
- This function also returns the `log_prob` and the `entropy` of the actions, which are necessary for calculating the PPO loss function in `train.py`.
