import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import numpy as np
import rb as replay_buffer
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter


class Q(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x


class DQN:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rbuffer: replay_buffer,
        action_set: torch.Tensor,
        writer: SummaryWriter,
        lr: float,
        epsilon: float,
        gamma: float,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = Q(input_dim, output_dim).to(self.device)
        self.q_target = Q(input_dim, output_dim).to(self.device)
        self._update_target()
        self.opt = Adam(self.q.parameters(), lr=lr)
        self.rbuffer = rbuffer
        self.action_set = action_set
        self.writer = writer
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_explore_episodes = 5
        self.update_frequency = 1000

    def _update_target(self):
        """Update target Q network with the current Q network"""
        for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
            target_param.data.copy_(param.data)

    def get_action_idx(self, state, ep):
        """Get action index based on epsilon-greedy policy"""
        # force early exploration
        if ep < self.num_explore_episodes:
            return ep % self.action_set.shape[0]

        # follow greedy policy with probability epsilon
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_set.shape[0])

        # follow policy when not training or (training and rand > self.epsilon)
        self.q.eval()
        with torch.no_grad():
            state = torch.tensor(state, device=self.device).float()
            q_values = self.q(state)
            action_idx = torch.argmax(q_values).item()
        self.q.train()
        return action_idx

    def train(self, batch: namedtuple, counter):
        """Train the Q network with a batch of transitions"""
        state = torch.stack([transition.state for transition in batch]).to(self.device)
        action = torch.tensor(
            [transition.action for transition in batch], device=self.device
        )
        reward = torch.tensor(
            [transition.reward for transition in batch], device=self.device
        )
        next_state = torch.stack([transition.next_state for transition in batch]).to(
            self.device
        )

        # calculate target
        self.q_target.eval()
        with torch.no_grad():
            target = (
                reward + self.gamma * torch.max(self.q_target(next_state), dim=1).values
            )
        self.q.train()
        self.q_target.train()

        # calculate loss
        q_values = self.q(state)
        q_values = q_values.gather(1, action.unsqueeze(1).long())
        loss = F.mse_loss(q_values, target.unsqueeze(1))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if counter % self.update_frequency == 0:
            self._update_target()
        return loss.item()

    def validate(self, state):
        """Validate the learned Q network"""
        self.q.eval()
        with torch.no_grad():
            state = torch.tensor(state, device=self.device).float()
            q_values = self.q(state)
            action_idx = torch.argmax(q_values).item()
        self.q.train()
        return action_idx
