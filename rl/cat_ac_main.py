import nn
import environment

import yaml
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def initialize_environment(config, update_type="local_ens"):
    args = environment.EnvArgs()
    for key, value in config.items():
        setattr(args, key, value)
    args.update_type = update_type
    target = np.loadtxt("snoopy.txt")
    target = torch.from_numpy(target).float()
    target = torch.nn.functional.interpolate(
        target.unsqueeze(0).unsqueeze(0),
        size=(args.n_side, args.n_side),
        mode="bilinear",
        align_corners=False,
    )
    target = target.squeeze()
    args.target_dist = target
    env = environment.KMCEnv(args)
    return env, target


def initialize_model(state_dim, action_dim, actions, writer):
    return nn.CategoricalA2C(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=actions,
        writer=writer,
        lr=0.00000001,
        gamma=0.99,
        lamb=0.95,
    )


def check_done(max_np, env, steps_per_move, c):
    if env.num_np > max_np or env.time > 1e6 or int(c + 1) * steps_per_move > 1e5:
        return torch.tensor(1)
    return torch.tensor(0)


def train_episode(env, model, writer, max_np, steps_per_move, batch_size, episode, counter):
    env.reset()
    state = env.sim_box[:, :, 1]
    done = False
    episode_reward = 0
    episode_rewards = []
    step = 0
    state_list, action_list, reward_list, next_state_list, done_list = (
        [],
        [],
        [],
        [],
        [],
    )

    while not done:
        action = model.choose_action(state)
        next_state, reward = env.step(steps_per_move, action, max_np)
        done = check_done(max_np, env, steps_per_move, step)
        state_list.append(state)
        action_list.append(action.cpu())
        reward_list.append(reward)
        next_state_list.append(next_state)
        done_list.append(done)
        state = next_state
        episode_reward += reward
        step += 1
        if len(state_list) >= batch_size or done:
            model.train(
                state_list, action_list, reward_list, next_state_list, done_list, counter
            )
            state_list, action_list, reward_list, next_state_list, done_list = (
                [],
                [],
                [],
                [],
                [],
            )

    episode_rewards.append(episode_reward)
    writer.add_scalar("total_episode_reward", episode_reward, episode)


def validate(env, model, writer, traj_name, eval_set, steps_per_move, max_np):
    state = env.reset()
    state = env.sim_box[:, :, 1]
    done = False
    env.sim.save_traj(f"{traj_name}_ep{eval_set}.xyz")
    step = 0
    episode_reward = 0

    while not done:
        action = model.choose_action(state)
        next_state, reward = env.step(steps_per_move, action, max_np)
        done = check_done(max_np, env, steps_per_move, step)
        state = next_state
        episode_reward += reward
        step += 1

    writer.add_scalar("cumulative_eval_episode_reward", episode_reward, eval_set)


def main():
    config = load_config("config.yaml")
    config["update_type"] = "local_ens"
    env, target = initialize_environment(config)
    writer = SummaryWriter()
    actions = torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).to(
        device='cuda'
    )
    model = initialize_model(env.sim_box.shape[0], env.sim_box.shape[0], actions, writer)

    traj_name = f"traj_{config["update_type"]}"
    max_np = 1000
    steps_per_move = 5
    batch_size = 16
    n_episodes = 500
    n_eval_sets = 10
    counter = 0

    for eval_sets in range(n_eval_sets):
        for episode in range(n_episodes):
            train_episode(
                env, model, writer, max_np, steps_per_move, batch_size, episode, counter
            )
            counter += 1

        validate(env, model, writer, traj_name, eval_sets, steps_per_move, max_np)
        model.save(f"model_{config["update_type"]}.pt")


if __name__ == "__main__":
    main()
