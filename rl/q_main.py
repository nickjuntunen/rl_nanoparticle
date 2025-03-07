import os
import nn
import rb as replay_buffer
import environment

import yaml
import torch
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser


# parse arguments
parser = ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--update_type", "-u", type=str, default="global_ens", choices=["temp", "global_ens"])
cl_args = parser.parse_args()

# initialize simulation environment
args = environment.EnvArgs()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
for key, value in config.items():
    setattr(args, key, value)

# set action spaces and targets
if args.update_type == "temp":
    actions = torch.linspace(0, 2, 11)[1:].view(-1, 1)
    action_dim = actions.shape[0]
    target = torch.tensor(1)
else:
    actions = torch.linspace(-args.enn * 1.2, args.enn * 1.2, int(4 * args.enn)).view(-1, 1)
    action_dim = actions.shape[0]
    target = torch.tensor(25)

traj_name = f"traj_{args.update_type}"
args.target_dist = target
args.update_type = cl_args.update_type
env = environment.KMCEnv(args)

# initialize replay buffer
rb = replay_buffer.ReplayBuffer(1000)

# initialize logger
writer = SummaryWriter()

# initialize Q-network
model = nn.DDQN(
    state_dim=env.state.shape[0],
    action_dim=action_dim,
    rbuffer=rb,
    action_set=actions,
    writer=writer,
    lr=0.00001,
    epsilon=0.1,
    gamma=0.99,
)

def check_done(max_np, env, steps_per_move, c):
    if env.num_np > max_np:
        return torch.tensor(1)
    if env.time > 1e6:
        return torch.tensor(1)
    if int (c+1)*steps_per_move > 1e5:
        return torch.tensor(1)
    return torch.tensor(0)

# train
n_episodes = 300
batch_size = 32
max_np = 1000
steps_per_move = 5
counter = 0
loss = 0
for eval_sets in range(5):
    # training step
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        c = 0
        while not done:
            act_idx = model.choose_action(state, episode)
            action = actions[act_idx]
            next_state, reward = env.step(steps_per_move, action, max_np, c)
            done = check_done(max_np, env, steps_per_move, c)
            rb.store(state, act_idx, reward, next_state, done)
            state = next_state
            if len(rb) > batch_size:
                loss = model.train(rb.sample(batch_size), counter)
                counter += 1
                c += 1

    # validation step
    print(f"Validation set {eval_sets}")
    state = env.reset()
    done = False
    env.sim.save_traj(traj_name + "_ep" + str(eval_sets) + ".xyz")
    c = 0
    while not done:
        act_idx = model.validate(state)
        action = actions[act_idx]
        next_state, reward = env.step(steps_per_move, action, c)
        env.sim.save_traj(traj_name + "_ep" + str(eval_sets) + ".xyz")
        state = next_state
        c += 1
        if env.num_np > max_np:
            dist = torch.sum(next_state)
            eval_loss = torch.nn.functional.mse_loss(dist, target)
            writer.add_scalar("eval_loss", eval_loss, eval_sets)
            done = True
        if env.time > 1e6:
            dist = torch.sum(next_state)
            eval_loss = torch.nn.functional.mse_loss(dist, target)
            writer.add_scalar("eval_loss", eval_loss, eval_sets)
            done = True
    # save model
    model.save(f"model_{args.update_type}.pt")

state = env.reset()
done = False
env.sim.save_traj(traj_name + "_final.xyz")
while not done:
    act_idx = model.validate(state)
    action = actions[act_idx]
    next_state, reward = env.step(steps_per_move, action)
    env.sim.save_traj(traj_name + "_final.xyz")
    state = next_state
    if env.num_np > max_np:
        dist = torch.sum(next_state)
        eval_loss = torch.nn.functional.mse_loss(dist, target)
        writer.add_scalar("eval_loss", eval_loss, eval_sets)
        done = True
# save model
model.save(f"model_{args.update_type}.pt")
