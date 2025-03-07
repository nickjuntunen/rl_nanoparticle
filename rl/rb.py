import torch
import random
from collections import namedtuple


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0
        self.transition = namedtuple(
            "Transition", ["state", "action", "reward", "next_state", "done"]
        )

    def store(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Store a transition in the replay buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = self.transition(state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> namedtuple:
        """Sample a batch of transitions from the replay buffer"""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
