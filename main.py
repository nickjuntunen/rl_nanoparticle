import os
import nn
import rb
import environment

import yaml
import torch
from argparse import ArgumentParser


# initialize simulation environment
args = environment.EnvArgs()
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
for key, value in config.items():
    setattr(args, key, value)
# set target_distribution
target = [0.0] * (config['n_side']**2 // 2)
target[100] = 25 # target: 50 clusters, each of size 100
args.target_dist = torch.tensor(target)
env = environment.KMCEnv(args)

# initialize replay buffer
rb = rb.ReplayBuffer(1000)

# initialize actions
if args.update_type == 'temp':
    actions = torch.linspace(0, 2, 11)[1:].view(-1,1)
    action_dim = actions.shape[0]
else:
    actions = torch.linspace(3, args.enn, 100).view(-1,1)
    action_dim = actions.shape[0]

# initialize Q model
state_dim = int(config['n_side']**2 / 2)
model = nn.QModel(state_dim=state_dim, action_dim=action_dim, rb=rb)

# train model
max_num_np = state_dim
max_ep = 5000
steps_per_action = 5
counter = 0
for ep in range(max_ep):
    state = env.reset().to(model.device)
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_times = []
    while env.num_np < max_num_np:
        episode_states.append(state)
        episode_times.append(env.time)
        print("state shape: ", state.shape)
        act_idx = model._get_action_idx(state)
        episode_actions.append(act_idx)
        env.step(steps_per_action, act_idx, actions)
        next_state, reward = env.get_state_reward()
        episode_rewards.append(reward)
        replay = [state, act_idx, reward, next_state, env.num_np >= max_num_np]
        rb.push(*replay)
        model.update_model(counter)
        print("next state shape: ", next_state.shape)
        state = next_state.to(model.device)
        print("after state shape: ", state.shape)
        counter += 1
