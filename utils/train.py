import json

import torch
from utils.evaluate import valid
from agent.ppo import PPO
from env.mdfjssp import MDFJSSPEnv
from env.base_class import *
from tqdm import tqdm


import os

def train(args, device):
    env = MDFJSSPEnv(args, None, device=device)
    agent = PPO(args, device=device)
    # rand_t = valid(args, device, agent)

    checkpoint_dir = args['save_path'] + args['checkpoint_id'] + '/'
    info_path = checkpoint_dir + 'info.json'
    checkpoint_path = checkpoint_dir + 'checkpoint.pth'

    start_iteration = 0
    best = 10000000
    step_left = 0
    makespan_list = []
    gain_list = []
    loss_list = []

    if os.path.exists(info_path) and os.path.exists(checkpoint_path):
        print("Found existing checkpoint, resuming training...")
        agent.load_checkpoint(checkpoint_dir)
        with open(info_path, 'r') as f:
            info = json.load(f)
        loss_list = info['loss']
        makespan_list = info['makespan']
        gain_list = info['gain']
        start_iteration = len(loss_list)
        if len(makespan_list) > 0:
            best = min(makespan_list)

    if args['early_stop'] is not None:
        step_left = args['early_stop']

    pbar = tqdm(total=args['iterations'], initial=start_iteration)
    for iteration in range(start_iteration, args['iterations']):
        if iteration % args['case_regen_iter'] == 0:
            # print("\n--------re-generate train cases--------\n")
            state = env.reset(keep_cases=False)
        else:
            state = env.reset(keep_cases=True)
        terminated = False
        memory = Memory()
        # print(f"iteration {iteration + 1}: collecting trajectory ...")
        with torch.no_grad():
            while not terminated:
                memory.add_state(state)
                scores, actions = agent.take_action(state)
                action_index, action_prop, runnable_cases = actions.get_action()
                state, reward, terminated, _, _ = env.step(actions)
                memory.add(reward, action_index, action_prop, runnable_cases)
        # print("update policy ...")
        loss = agent.update(memory)
        memory.clear()
        v_t, gain, _, _ = valid(args, device, agent)
        loss_list.append(loss)
        makespan_list.append(v_t)
        gain_list.append(gain)
        # early stop
        if v_t < best:
            best = v_t
            agent.save_policy(args['save_path'] + args['checkpoint_id'] + '/')
            if args['early_stop'] is not None:
                step_left = args['early_stop']
        else:
            step_left -= 1
            if args['early_stop'] is not None and step_left <= 0:
                break

        pbar.update(1)
        pbar.set_postfix(loss=loss, makespan=v_t, best=best)
        info = {'loss': loss_list, 'makespan': makespan_list, 'gain': gain_list}
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        json.dump(info, open(info_path, 'w'))
        agent.save_checkpoint(checkpoint_dir)
    pbar.close()
    print('finished')
