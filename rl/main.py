import os
import nn
import rb
import environment

import yaml
import torch
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter


# initialize simulation environment
args = environment.EnvArgs()
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
for key, value in config.items():
    setattr(args, key, value)
# set target_distribution
# target = [0.0] * (config['n_side']**2 // 2)
# target[100] = 25 # target: 50 clusters, each of size 100
# target a certain number of clusters
target = 50
args.target = torch.tensor(target)
env = environment.KMCEnv(args)

# initialize replay buffer
rb = rb.ReplayBuffer(1000)

# initialize actions
if args.update_type == 'temp':
    actions = torch.linspace(0, 2, 11)[1:].view(-1,1)
    action_dim = actions.shape[0]
else:
    actions = torch.linspace(-args.enn, args.enn, int(4*args.enn)).view(-1,1)
    action_dim = actions.shape[0]

# initialize logger
writer = SummaryWriter()

# initialize Q-network
model = nn.DQN(
    input_dim=env.state.shape[0],
    output_dim=action_dim,
    rbuffer=rb,
    action_set=actions,
    writer=writer,
    lr=0.001,
    epsilon=0.1,
    gamma=0.99
)


# train
n_episodes = 500
batch_size = 32
max_np = 1000
steps_per_move = 5
counter = 0
loss = 0
for eval_sets in range(10):
    # training step
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            act_idx = model.get_action_idx(state, episode)
            action = actions[act_idx]
            next_state, reward = env.step(5, action)
            rb.store(state, act_idx, reward, next_state)
            state = next_state
            if env.num_np > max_np:
                done = True
            if len(rb) > batch_size:
                loss = model.train(rb.sample(batch_size), counter)
            if env.time > 1e6:
                break
            counter += 1
        writer.add_scalar('reward', reward, episode)
        writer.add_scalar('num_np', env.num_np, episode)
        writer.add_scalar('time', env.time, episode)
        writer.add_scalar('action', action, episode)
        writer.add_scalar('loss', loss, episode)
    # validation step
    state = env.reset()
    done = False
    while not done:
        act_idx = model.get_action_idx(state, episode)
        action = actions[act_idx]
        next_state, reward = env.step(5, action)
        state = next_state
        if env.num_np > max_np:
            dist = torch.sum(next_state)
            eval_loss = torch.nn.functional.mse_loss(dist, target)
            writer.add_scalar('eval_loss', eval_loss, eval_sets)
            done = True
        if env.time > 1e6:
            dist = torch.sum(next_state)
            eval_loss = torch.nn.functional.mse_loss(dist, target)
            writer.add_scalar('eval_loss', eval_loss, eval_sets)
            break