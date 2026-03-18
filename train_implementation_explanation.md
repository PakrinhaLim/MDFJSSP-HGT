# MDFJSSP `train.py` Implementation Overview

The `train.py` script contains the main reinforcement learning (RL) training loop for the Multi-faceted Dynamic Flexible Job Shop Scheduling Problem (MDFJSSP). It uses **Proximal Policy Optimization (PPO)** to train the agent by interacting with the `MDFJSSPEnv` environment.

## 1. Initialization
```python
def train(args, device):
    env = MDFJSSPEnv(args, None, device=device)
    agent = PPO(args, device=device)
    ...
```
- It starts by initializing the environment (`MDFJSSPEnv`) and the RL agent (`PPO`). 
- It sets up trackers for the best makespan (`best = 10000000`), a counter for early stopping (`step_left`), and lists to log standard training metrics: loss, makespan, and gain.

## 2. The Main Training Loop & Environment Reset
```python
    pbar = tqdm(total=args['iterations'])
    for iteration in range(args['iterations']):
        if iteration % args['case_regen_iter'] == 0:
            state = env.reset(keep_cases=False)
        else:
            state = env.reset(keep_cases=True)
```
- It iterates for a specified number of `iterations`.
- **Case Generation Strategy**: To prevent the model from simply memorizing specific scheduling instances (overfitting), the environment occasionally generates entirely new problem instances (when `iteration % args['case_regen_iter'] == 0`). During the iterations in between, the agent practices on the exact same set of cases (`keep_cases=True`).

## 3. Trajectory Collection (Rollout Phase)
```python
        terminated = False
        memory = Memory()
        with torch.no_grad():
            while not terminated:
                memory.add_state(state)
                scores, actions = agent.take_action(state)
                action_index, action_prop, runnable_cases = actions.get_action()
                state, reward, terminated, _, _ = env.step(actions)
                memory.add(reward, action_index, action_prop, runnable_cases)
```
- The `Memory()` object acts as a buffer to store the states, actions, probabilities, and rewards.
- `with torch.no_grad():` ensures that PyTorch doesn't track gradients during the environment interaction, which speeds up simulation and saves memory.
- The `while not terminated:` loop continues until all active scheduling cases in the environment are completed.
- At each timestep:
  1. The agent examines the current environment `state` and predicts probability scores for actions, then samples an action.
  2. The chosen action is applied to the environment (`env.step`).
  3. The environment computes the `reward`, moves to the next `state`, and determines if all cases have `terminated`.
  4. The selected action, its network probability, and the reward are stored in memory.

## 4. Policy Update
```python
        loss = agent.update(memory)
        memory.clear()
```
- Once the rollout terminates, the accumulated memory is passed into `agent.update(memory)`.
- Behind the scenes (in `agent/ppo.py`), PPO computes the **Advantages** (which evaluate how much better an action was compared to the average expected action) and updates the actor-critic network (which is an `HGAN` model in this repository).
- `memory.clear()` empties the buffer for the next iteration.

## 5. Validation and Evaluation
```python
        v_t, gain, _ = valid(args, device, agent)
        loss_list.append(loss)
        makespan_list.append(v_t)
        gain_list.append(gain)
```
- The newly updated agent is tested immediately via the `valid` function on a separate set of validation instances. This returns the current makespan (`v_t`) and performance gain. 
- The metrics are appended to the tracking lists.

## 6. Model Checkpointing & Early Stopping
```python
        if v_t < best:
            best = v_t
            agent.save_policy(args['save_path'] + args['checkpoint_id'] + '/')
            if args['early_stop'] is not None:
                step_left = args['early_stop']
        else:
            step_left -= 1
            if args['early_stop'] is not None and step_left <= 0:
                break
```
- The algorithm tracks the `best` validation makespan seen so far.
- If the current validation makespan (`v_t`) is lower (better) than the historical `best`, it saves the neural network parameters locally to `args['save_path']` and resets the early stopping patience counter.
- If the model doesn't improve for `args['early_stop']` consecutive iterations, training is aborted early (`break`) to save time and compute resources.

## 7. Progress Logging
```python
        pbar.update(1)
        pbar.set_postfix(loss=loss, makespan=v_t, best=best)
        info = {'loss': loss_list, 'makespan': makespan_list, 'gain': gain_list}
        json.dump(info, open(args['save_path'] + args['checkpoint_id'] + '/' + 'info.json', 'w'))
```
- It logs the loss and makespan on your terminal progress bar (`tqdm`).
- Every iteration, it dumps the tracked metrics into an `info.json` file inside the model's checkpoint folder so that you can later plot the learning curves.
