import nn
import environment
import multiprocess
import rb as replay_buffer

import yaml
import time
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
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--target_s", "-t", type=int, default=25)
    return parser.parse_args()


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)
    

def main():
    args = parse_arguments()
    config = load_config(args.config)
    seed = args.seed
    train_loops_per_episode = int(config["train_loops_per_episode"])
    n_episodes = int(config["n_episodes"])
    n_training_sets = int(config["n_training_sets"])
    batch_size = int(config["batch_size"])
    target_experiences = int(30000 * float(config["num_workers"]) / float(config["steps_per_move"]))
    model_state_dict_path = f"./global_ens/model_{args.update_type}_latest_state_dict.pt"

    # Initialize environment, replay buffer, writer in main process for model setup
    env, actions, target = environment.initialize_environment(config, args.seed)
    rb = replay_buffer.ReplayBuffer(500000)
    writer = SummaryWriter()
    model = nn.DDQN(
        state_dim=env.state.shape[0],
        action_dim=actions.shape[0],
        rbuffer=rb,
        action_set=actions,
        writer=writer,
        lr=0.001,
        epsilon=0.1,
        gamma=0.99,
    )
    
    # Save initial model state dict (not the whole model)
    torch.save(model.state_dict(), model_state_dict_path)

    ctx = multiprocess.MPContext(max_queue_size=target_experiences)

    train_steps = 0
    for train_set in range(n_training_sets):
        ctx.create_workers(config, model_state_dict_path)
        print(f"Training set {train_set + 1}/{n_training_sets}")
        for ep in range(n_episodes):
            print(f"Episode {ep + 1}/{n_episodes}")
            ctx.assign_episode(ep)
            while not (ctx.end_queue.full() and ctx.experience_queue.empty()):
                time.sleep(5)
                ctx.process_experiences(model, rb)
                try:
                    for _ in range(train_loops_per_episode):
                        batch = rb.sample(batch_size)
                        model.train(batch, train_steps)
                        train_steps += 1
                except:
                    pass

        ctx.stop_workers()
        model.save(f"./global_ens/{train_set}_model.pt")
        model.save(model_state_dict_path)

    ctx.close_workers()
    writer.close()
    # Save the final model state dict
    torch.save(model.state_dict(), model_state_dict_path)
    print(f"Final model state dict saved to {model_state_dict_path}")


if __name__ == "__main__":
    main()
