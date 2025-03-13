import nn
import environment
import rb as replay_buffer

import yaml
import torch
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--update_type",
        "-u",
        type=str,
        default="global_ens",
        choices=["temp", "global_ens"],
    )
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def initialize_environment(config, update_type):
    args = environment.EnvArgs()
    for key, value in config.items():
        setattr(args, key, value)
    args.update_type = update_type

    if update_type == "temp":
        actions = torch.logspace(-2, 0.5, 11)[1:].view(-1, 1)
        target_cluster = torch.tensor(1)
        target_area = torch.tensor(1.0)
        target = (target_cluster, target_area)
    else:
        actions = torch.linspace(
            -args.enn * 1.2, args.enn * 1.2, int(4 * args.enn)
        ).view(-1, 1)
        target_cluster = torch.tensor(25)
        target_area = torch.tensor(0.15)
        target = (target_cluster, target_area)

    args.target_dist = target
    env = environment.KMCEnv(args)
    return env, actions, target


def initialize_model(env, action_dim, actions, rb, writer):
    return nn.DDQN(
        state_dim=env.state.shape[0],
        action_dim=action_dim,
        rbuffer=rb,
        action_set=actions,
        writer=writer,
        lr=0.000001,
        epsilon=0.1,
        gamma=0.99,
    )


def check_done(max_np, env, steps_per_move, c):
    if env.num_np > max_np or env.time > 1e6 or int(c + 1) * steps_per_move > 1e5:
        return torch.tensor(1)
    return torch.tensor(0)


def train_episode(
    env, model, rb, actions, max_np, steps_per_move, batch_size, episode, counter
):
    state = env.reset()
    done = False
    loss = 0
    c = 0
    while not done:
        act_idx = model.choose_action(state, episode)
        action = actions[act_idx]
        next_state, reward = env.step(steps_per_move, action, max_np)
        done = check_done(max_np, env, steps_per_move, c)
        rb.store(state, act_idx, reward, next_state, done)
        state = next_state
        if len(rb) > batch_size:
            loss = model.train(rb.sample(batch_size), counter)
            counter += 1
            c += 1
    return counter, loss


def validate(
    env, model, actions, writer, traj_name, eval_set, steps_per_move, max_np
):
    state = env.reset()
    done = False
    env.sim.save_traj(f"{traj_name}_ep{eval_set}.xyz")
    c = 0
    cumulative_rewards = 0
    while not done:
        act_idx = model.validate(state)
        action = actions[act_idx]
        next_state, reward = env.step(steps_per_move, action, max_np)
        cumulative_rewards += reward
        env.sim.save_traj(f"{traj_name}_ep{eval_set}.xyz")
        state = next_state
        c += 1
        writer.add_scalar(f"eval_set_{eval_set}_actions", action, c)
        writer.add_scalar(f"eval_set_{eval_set}_cum_rewards", cumulative_rewards, c)
        if env.num_np > max_np or env.time > 1e6:
            # dist = torch.sum(next_state)
            # eval_loss = torch.nn.functional.mse_loss(dist, target)
            # writer.add_scalar("eval_loss", eval_loss, eval_set)
            done = True


def main():
    args = parse_arguments()
    config = load_config(args.config)
    env, actions, target = initialize_environment(config, args.update_type)
    rb = replay_buffer.ReplayBuffer(100000)
    writer = SummaryWriter()
    model = initialize_model(env, actions.shape[0], actions, rb, writer)

    traj_name = f"traj_{args.update_type}"
    n_episodes = 300
    n_eval_sets = 5
    batch_size = 32
    max_np = 1000
    steps_per_move = 3
    counter = 0

    for eval_sets in range(n_eval_sets):
        for episode in range(n_episodes):
            counter, loss = train_episode(
                env,
                model,
                rb,
                actions,
                max_np,
                steps_per_move,
                batch_size,
                episode,
                counter,
            )

        print(f"Validation set {eval_sets}")
        validate(
            env,
            model,
            actions,
            writer,
            traj_name,
            eval_sets,
            steps_per_move,
            max_np,
        )
        model.save(f"model_{args.update_type}.pt")

    validate(
        env, model, actions, writer, traj_name, "final", steps_per_move, max_np
    )


if __name__ == "__main__":
    main()
