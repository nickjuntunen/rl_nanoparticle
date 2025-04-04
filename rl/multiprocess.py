import torch.multiprocessing as mp
import torch
import queue
import time
import gc

mp.set_start_method("spawn", force=True)

import environment
import nn


def check_sim_done(max_np, max_moves, max_time, env, steps_per_move, c):
    if env.num_np >= max_np or int(c) * steps_per_move > max_moves or env.time > max_time:
        return torch.tensor(1)
    return torch.tensor(0)


def worker_function(worker_id, episode_queues, experience_queue, end_queue, config, model_path):
    """
    The function that each worker process will run.
    """
    # worker initialization
    episode_queue = episode_queues[worker_id]
    seed = config["seed"]
    env, actions, _ = environment.initialize_environment(
        config,
        seed + worker_id
    )
    state_dim = env.state.shape[0]
    action_shape = actions.shape[0]
    model = nn.DDQN(
        state_dim=state_dim,
        action_dim=action_shape,
        rbuffer=None,
        action_set=actions,
        writer=None,
        lr=0.001,
        epsilon=0.1,
        gamma=0.99,
    )
    
    # start the worker loop
    while True:
        # wait for each episode to be assigned
        try:
            # block until an episode is available
            entry = episode_queue.get()
            if entry == "STOP":
                print(f"Worker {worker_id} received STOP signal. End simulation.")
                break

            elif isinstance(entry, int):
                episode = entry
                model.load_state_dict(torch.load(model_path))
                model.eval()
                run_episode(
                    worker_id,
                    episode,
                    env,
                    model,
                    actions,
                    experience_queue,
                    config
                )
                end_queue.put(worker_id)

            else:
                # error handling
                print(f"Worker {worker_id} received invalid entry: {entry}.")
                # skip to the next episode message
                continue

        except Exception as e:
            print(f"Worker {worker_id} encountered an error: {e}")
            time.sleep(1)
            continue


def run_episode(id, episode, env, model, actions, experience_queue, config):
    # variable initialization
    max_np = int(config["max_np"])
    max_moves = int(config["max_moves"])
    max_time = int(config["max_time"])
    steps_per_move = int(config["steps_per_move"])
    state = env.reset()
    done = False
    it = 0

    # run the episode
    while not done:
        # make sure there is space in the experience queue
        if experience_queue.full():
            print("Experience queue is full. Waiting for space...")
            time.sleep(4)
            continue
        
        act_idx = model.choose_action(state, episode)
        action = actions[act_idx]
        next_state, reward = env.step(
            steps_per_move,
            action
        )
        done = check_sim_done(
            max_np,
            max_moves,
            max_time,
            env,
            steps_per_move,
            it
        )
        experience = {
            'state': state.clone().cpu().numpy(),
            'action': act_idx,
            'reward': reward,
            'next_state': next_state.clone().cpu().numpy(),
            'done': done
        }
        while experience_queue.full():
            print("Experience queue is full (storing step). Waiting for space...")
            time.sleep(4)
        
        # attempt to send to main process (trainer)
        put_successful = False
        retry_count = 0
        while not put_successful and retry_count < 5:
            try:
                experience_queue.put(experience, timeout=1.0)
                put_successful = True
            except queue.Full:
                print("Experience queue is full (storing step). Waiting for space...")
                retry_count += 1
                time.sleep(4)

        if not put_successful:
            print(f"Worker {id} failed to put experience after 5 attempts. Skipping this step.")
            
        state = next_state
        it += 1
        if it % 200 == 0:
            gc.collect()

    print(f"Worker {id} completed episode {episode} after {it} iterations.")


class MPContext:
    """
    A context manager for managing the multiprocessing context.
    """
    def __init__(self, max_queue_size=10):
        self.max_queue_size = max_queue_size
        self.ctx = mp.get_context("spawn")
        self.episode_queues = None
        self.experience_queue = self.ctx.Queue(maxsize=max_queue_size)
        self.end_queue = None
        self.workers = None
        self.num_workers = None
        self.processes = None

    def create_workers(self, config, model_path):
        self.num_workers = config["num_workers"]
        if self.num_workers <= 0:
            raise ValueError("Number of workers must be greater than 0.")
        if self.num_workers > mp.cpu_count():
            raise ValueError(f"Number of workers exceeds available CPU cores: {mp.cpu_count()}.")
        self.end_queue = self.ctx.Queue(maxsize=self.num_workers)
        self.episode_queues = [self.ctx.Queue(maxsize=self.max_queue_size) for _ in range(self.num_workers)]
        self.processes = []
        for worker_id in range(self.num_workers):
            p = self.ctx.Process(
                target=worker_function,
                args=(
                    worker_id,
                    self.episode_queues,
                    self.experience_queue,
                    self.end_queue,
                    config,
                    model_path,
                )
            )
            self.processes.append(p)
            p.start()

    def stop_workers(self):
        """Send stop signal to all workers"""
        for q in self.episode_queues:
            q.put("STOP")

        for p in self.processes:
            p.join(timeout=5)
        
    def cleanup(self):
        # Wait for processes to finish
        for p in self.processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

    def assign_episode(self, episode):
        # empty the end_queue
        while not self.end_queue.empty():
            try:
                self.end_queue.get_nowait()
            except queue.Empty:
                break
        # assign the episode to each worker
        for q in self.episode_queues:
            q.put(episode)


    def reset_queues(self):
        self.experience_queue = self.ctx.Queue(maxsize=self.max_queue_size)
        self.end_queue = self.ctx.Queue(maxsize=self.num_workers)
        self.episode_queues = [self.ctx.Queue(maxsize=self.max_queue_size) for _ in range(self.num_workers)]
        print("Queues have been reset.")

    
    def process_experiences(self, model, rb, max_batch=100):
        processed = 0
        while not self.experience_queue.empty():
            batch_processed = 0
            for _ in range(max_batch):
                try:
                    experience = self.experience_queue.get(block=False)
                    state = torch.tensor(experience['state'], dtype=torch.float32)
                    action = experience['action']
                    reward = experience['reward']
                    next_state = torch.tensor(experience['next_state'], dtype=torch.float32)
                    done = experience['done']
                    rb.store(state, action, reward, next_state, done)
                    batch_processed += 1

                    # clean up references
                    del state, reward, next_state, done, experience

                except queue.Empty:
                    break
            
            processed += batch_processed
            if batch_processed == max_batch:
                gc.collect()
                time.sleep(0.05)

        gc.collect()
        print(f"Processed {processed} experiences from the queue.")
        return processed
