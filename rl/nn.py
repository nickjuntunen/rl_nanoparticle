import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import numpy as np
import rb as replay_buffer
from policy import GaussianActorCriticPolicy, CategoricalActorCriticPolicy
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

    def choose_action(self, state, ep=None):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class Q(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
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
        actions = torch.tensor([t.action for t in batch], device=self.device).view(
            -1, 1
        )
        rewards = torch.tensor([t.reward for t in batch], device=self.device).view(
            -1, 1
        )
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
            targets = rewards + (1 - dones) * self.gamma * next_q_targets

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


class ActorCriticNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        policy_type: str = "gaussian",
        action_space_len: int = 1,
    ):
        super(ActorCriticNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, 1, kernel_size=5, padding=2),
            nn.Tanh(),
        )

        if policy_type == "categorical":
            self.actor_head = nn.Linear(
                self.action_dim**2, self.action_dim**2 * action_space_len
            )
            self.actor_log_std = nn.Parameter(
                torch.zeros(1, self.action_dim**2 * action_space_len)
            )
        else:
            self.actor_head = nn.Linear(self.action_dim**2, self.action_dim**2)
            self.actor_log_std = nn.Parameter(torch.zeros(1, self.action_dim**2))

        nn.init.uniform_(self.actor_head.weight, 3.0, 5.0)
        nn.init.uniform_(self.actor_head.bias, 3.0, 5.0)
        self.critic_head = nn.Linear(self.action_dim**2, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        action_mean = self.actor_head(x)
        action_mean = action_mean.view(-1, self.action_dim, self.action_dim)
        action_log_std = self.actor_log_std.view(-1, self.action_dim, self.action_dim)
        action_std = torch.exp(action_log_std)
        value = self.critic_head(x)
        return action_mean, action_std, value


class A2C(LearningArchitecture):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
    ):
        super(A2C, self).__init__(state_dim, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = None
        self.gamma = None
        self.lamb = None

    def save(self, path):
        torch.save(self.ac.state_dict(), path)
        return

    def _calculate_advantage(self, rewards, values, next_values, dones):
        advantages = []
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            returns.insert(0, rewards[i] + self.gamma * next_values[i] * (1 - dones[i]))
            delta = returns[0] - values[i]
            gae = delta + self.gamma * self.lamb * gae
            advantages.insert(0, gae)
        return advantages, returns

    def train(self, state, actions, rewards, next_state, dones, counter):
        state = torch.stack(state).to(self.device).float()
        actions = torch.tensor(np.array(actions), device=self.device).float()
        rewards = torch.tensor(np.array(rewards), device=self.device).float()
        next_state = torch.tensor(np.array(next_state), device=self.device).float()
        dones = torch.tensor(np.array(dones), device=self.device).float()

        _, _, values = self.ac(state)
        _, _, next_values = self.ac(next_state)

        advantage, returns = self._calculate_advantage(
            rewards, values, next_values, dones
        )
        advantage = torch.stack([a.clone() for a in advantage]).to(self.device)
        returns = torch.stack([r.clone() for r in returns]).to(self.device)

        self._update_actor_critic(state, actions, values, returns, advantage, counter)
        return

    def choose_action(self, state, ep=None):
        return super().choose_action(state, ep)

    def _update_actor_critic(self, state, actions, values, returns, advantage, counter):
        raise NotImplementedError


class GaussianA2C(A2C):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        writer: SummaryWriter,
        lr: float,
        gamma: float,
        lamb: float,
    ):
        super(GaussianA2C, self).__init__(state_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCriticNet(state_dim, action_dim).to(self.device)
        self.opt_a = Adam(self.ac.parameters(), lr=lr)
        self.scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_a, mode="min", factor=0.5, patience=10
        )
        self.opt_c = Adam(self.ac.parameters(), lr=lr)
        self.scheduler_c = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_c, mode="min", factor=0.5, patience=10
        )
        self.policy = GaussianActorCriticPolicy(self.ac)
        self.writer = writer
        self.gamma = gamma
        self.lamb = lamb

    def choose_action(self, state):
        state = state.unsqueeze(0).to(self.device).float()
        action = self.policy.act(state, False)
        action = torch.clamp(action, -10, 10)
        return action

    def _update_actor_critic(self, state, actions, values, returns, advantage, counter):
        critic_loss = F.mse_loss(values, returns.detach())
        distribution = self.policy.action_distribution(state)
        log_probs = distribution.log_prob(actions)
        # entropy = distribution.entropy().mean()
        # norm_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        actor_loss = -torch.clamp(
            (log_probs * advantage).mean(), -0.2, 0.2
        )  # - 0.1 * entropy
        loss = actor_loss + critic_loss

        self.opt_a.zero_grad()
        self.opt_c.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=1.0)
        self.opt_a.step()
        self.opt_c.step()

        self.writer.add_scalar("actor loss", actor_loss.item(), counter)
        self.writer.add_scalar("critic loss", critic_loss.item(), counter)
        return


class CategoricalA2C(A2C):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_space: torch.tensor,
        writer: SummaryWriter,
        a_lr: float,
        c_lr: float,
        gamma: float,
        lamb: float,
    ):
        super(CategoricalA2C, self).__init__(state_dim, action_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ac = ActorCriticNet(
            state_dim,
            action_dim,
            policy_type="categorical",
            action_space_len=len(action_space),
        ).to(self.device)
        self.opt_a = Adam(self.ac.parameters(), lr=a_lr)
        self.scheduler_a = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_a, mode="min", factor=0.5, patience=10
        )
        self.opt_c = Adam(self.ac.parameters(), lr=c_lr)
        self.scheduler_c = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_c, mode="min", factor=0.5, patience=10
        )
        self.action_space = action_space
        self.policy = CategoricalActorCriticPolicy(self.ac, action_space)
        self.writer = writer
        self.gamma = gamma
        self.lamb = lamb

    def choose_action(self, state, ep=None):
        state = state.unsqueeze(0).to(self.device).float()
        action = self.policy.act(state, False)
        return action

    def _update_actor_critic(self, state, actions, values, returns, advantage, counter):
        critic_loss = F.mse_loss(values, returns.detach())

        distribution = self.policy.action_distribution(state)
        action_indices = (
            (self.policy.action_space.unsqueeze(0) - actions.unsqueeze(-1)).abs() < 1e-6
        ).nonzero()[:, -1]
        log_probs = distribution.log_prob(action_indices)
        entropy = distribution.entropy().mean()

        # ratio = torch.exp(current_log_probs - log_probs.flatten())
        # surr1 = ratio * advantage
        # surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage
        # actor_loss = -torch.min(surr1, surr2).mean()
        # actor_loss -= 0.01 * entropy

        actor_loss = -torch.clamp(
            (log_probs * advantage).mean(), -0.2, 0.2
        ) - 0.1 * entropy

        loss = actor_loss + 0.5 * critic_loss

        self.opt_a.zero_grad()
        self.opt_c.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
        self.opt_a.step()
        self.opt_c.step()

        self.writer.add_scalar("actor loss", actor_loss.item(), counter)
        self.writer.add_scalar("critic loss", critic_loss.item(), counter)
        self.writer.add_scalar("entropy", entropy.item(), counter)

        return
