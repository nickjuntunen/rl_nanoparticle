import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import numpy as np
import rb as replay_buffer
from policy import ActorCriticPolicy
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter

torch.autograd.set_detect_anomaly(True)

class LearningArchitecture:
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def train(self, x):
        raise NotImplementedError

    def validate(self, x):
        raise NotImplementedError

    def choose_action(self, state, ep):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class Q(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x


class DDQN(LearningArchitecture):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rbuffer: replay_buffer,
        action_set: torch.Tensor,
        writer: SummaryWriter,
        lr: float,
        epsilon: float,
        gamma: float,
    ):
        super(DDQN, self).__init__(state_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q = Q(state_dim, action_dim).to(self.device)
        self.q_target = Q(state_dim, action_dim).to(self.device)
        self._update_target()
        self.opt = Adam(self.q.parameters(), lr=lr)
        self.rbuffer = rbuffer
        self.action_set = action_set
        self.writer = writer
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_explore_episodes = 5
        self.update_frequency = 100

    def _update_target(self):
        """Update target Q network with the current Q network"""
        for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
            target_param.data.copy_(param.data)

    def choose_action(self, state, ep):
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
            state = state.to(self.device).float()
            q_values = self.q(state)
            action_idx = torch.argmax(q_values).item()
        self.q.train()
        return action_idx

    def train(self, batch: namedtuple, counter):
        """Train the Q network with a batch of transitions"""
        states = torch.stack([t.state for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], device=self.device).view(-1, 1)
        rewards = torch.tensor([t.reward for t in batch], device=self.device).view(-1, 1)
        next_states = torch.stack([t.next_state for t in batch]).to(self.device)
        dones = torch.stack([t.done for t in batch]).to(self.device).view(-1, 1)

        self.q.eval()
        with torch.no_grad():
            next_action_values = self.q(next_states)
            next_actions = torch.max(next_action_values, dim=1).indices
            # max_next_q_values = torch.max(next_q_values, dim=1).values
            # target = rewards + self.gamma * max_next_q_values.unsqueeze(-1)

        # self.q.train()
        self.q_target.eval()
        with torch.no_grad():
            next_q_targets = self.q_target(next_states)
            next_q_targets = torch.gather(next_q_targets, 1, next_actions.unsqueeze(-1))
            targets = rewards + (1-dones) * self.gamma * next_q_targets
        
        self.q.train()
        q_values = self.q(states)
        q_values = torch.gather(q_values, 1, actions)

        loss = F.mse_loss(q_values, targets)

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 1)
        self.opt.step()

        if counter % self.update_frequency == 0:
            self._update_target()
        
        self._log_metrics(q_values, rewards, counter)

        return loss.item()

    def _log_metrics(self, q_values, reward, counter):
        self.writer.add_scalar("max q-value", torch.max(q_values).item(), counter)
        self.writer.add_scalar("min q-value", torch.min(q_values).item(), counter)
        self.writer.add_scalar("reward", torch.mean(reward).item(), counter)
        self.writer.add_scalar("max reward", torch.max(reward).item(), counter)
        self.writer.add_scalar("min reward", torch.min(reward).item(), counter)

    def validate(self, state):
        """Validate the learned Q network"""
        self._update_target()
        self.q_target.eval()
        with torch.no_grad():
            state = state.to(self.device).float()
            q_values = self.q_target(state)
            action_idx = torch.argmax(q_values).item()
        return action_idx

    def save(self, path):
        torch.save(self.q.state_dict(), path)


class ActorNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(ActorNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(state_dim**2, 256)
        self.fc2 = nn.Linear(256, 128)

        self.actor_mean = nn.Linear(128, state_dim**2)
        self.actor_log_std = nn.Parameter(torch.zeros(1, state_dim**2))


    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten the CNN output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        action_std = action_std.view(-1, 1, self.state_dim, self.action_dim)
        action_mean = action_mean.view(-1, 1, self.state_dim, self.action_dim)
        return action_mean, action_std
    

class CriticNet(nn.Module):
    def __init__(self, state_dim: int):
        super(CriticNet, self).__init__()
        self.input_dim = state_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(state_dim**2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class A2C(LearningArchitecture):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        writer: SummaryWriter,
        lr: float,
        gamma: float,
        lamb: float
    ):
        super(A2C, self).__init__(state_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNet(state_dim, action_dim).to(self.device)
        self.critic = CriticNet(state_dim).to(self.device)
        self.opt_a = Adam(self.actor.parameters(), lr=lr)
        self.opt_c = Adam(self.critic.parameters(), lr=lr)
        self.policy = ActorCriticPolicy(self.actor, self.critic)
        self.writer = writer
        self.gamma = gamma
        self.lamb = lamb

    def choose_action(self, state):
        state = state.unsqueeze(0).to(self.device).float()
        action = self.policy.act(state)
        action = action.clamp(-10, 10)
        return action
    
    def _calculate_advantage(self, rewards, values, next_values, dones):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lamb * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
        return returns
    
    def train(self, state, actions, rewards, next_state, dones, counter):
        state = torch.tensor(np.array(state), device=self.device).float()
        actions = torch.tensor(np.array(actions), device=self.device).float()
        rewards = torch.tensor(np.array(rewards), device=self.device).float()
        next_state = torch.tensor(np.array(next_state), device=self.device).float()
        dones = torch.tensor(np.array(dones), device=self.device).float()

        values = self.critic(state)
        next_values = self.critic(next_state)
        advantage_c = self._calculate_advantage(rewards, values, next_values, dones)
        advantage_c = torch.stack([a.clone() for a in advantage_c]).to(self.device)
        advantage_a = torch.clone(advantage_c).detach()
        
        self._update_critic(state, advantage_c, counter)
        self._update_actor(state, actions, advantage_a, counter)
        return

    def _update_critic(self, state, advantage, counter):
        values = self.critic(state)
        loss = F.mse_loss(values, advantage)
        self.opt_c.zero_grad()
        loss.backward(retain_graph=True)
        self.opt_c.step()
        self.writer.add_scalar("critic loss", loss.item(), counter)
        return
    
    def _update_actor(self, state, actions, advantage, counter):
        state = state.unsqueeze(1)
        distribution = self.policy.action_distribution(state)
        actions = actions.view(-1, self.state_dim, self.action_dim)
        log_probs = distribution.log_prob(actions)
        loss = -(log_probs * advantage).mean()
        self.opt_a.zero_grad()
        loss.backward()
        self.opt_a.step()
        self.writer.add_scalar("actor loss", loss.item(), counter)
        return

    def validate(self, state):
        raise NotImplementedError

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
        return
