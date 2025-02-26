import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
import numpy as np
import warnings
import rb as replay_buffer


class Q(nn.Module):
    def __init__(self, input_dim, output_dim):
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


class QModel:
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        rb : replay_buffer.ReplayBuffer,
        tau = 0.005,
        # out_scaling = 80,
        device = torch.device('cuda:0')
    ):
        '''
        Arguments:
            state_dim {int} -- dimension of state
            action_dim {int} -- dimension of action
            tau {float} -- tau for soft update
            # out_scaling {float} -- amount to scale output of network by
            device {torch.device} -- device to use for network
        '''
        self.device = device
        self.tau = tau
        self.gamma = 0.9
        self.epsilon = 0.1
        self.batch_size = 32
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.rb = rb
        self.q = Q(state_dim, action_dim).to(device)
        self.q_target = Q(state_dim, action_dim).to(device)
        self.q_opt = Adam(self.q.parameters(), lr=0.0001)
        self._update(self.q_target, self.q)
        self.loss = nn.KLDivLoss('batchmean')

    def _update(self, target, source):
        ''' Soft update target network with source network
        '''
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def update_target(self):
        ''' Update target network
        '''
        self._update(self.q_target, self.q)

    def update_model(self, epoch):
        ''' Update model
        '''
        if len(self.rb) < self.batch_size:
            return
        assert self.q.training
        assert not self.q_target.training

        transitions = self.rb.sample(self.batch_size)
        batch = self.rb.transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        
        with torch.no_grad():
            q_next = ((~done_batch) * (self.q_target(next_state_batch).min(dim=1)[0]))
            q_target = reward_batch + self.gamma * q_next

        q_values = self.q(state_batch)
        q_values = q_values.gather(1, action_batch)
        loss = self.loss(q_values, q_target)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()

        # get estimate for optimal Q function
        last_exp = self.rb.get_element_from_buffer(self.rb.position-1)
        last_state = last_exp.state
        self.q.eval()
        with torch.no_grad():
            last_state = torch.tensor(last_state).to(self.device)
            q_val = torch.min(self.q(last_state))
        self.q.train()
        self.writer.add_scalar('Q value', q_val.item(), epoch)
        self.writer.add_scalar('Loss', loss.item(), epoch)

    def save_network(self, path):
        ''' Save network to path
        '''
        torch.save({'model_state_dict': self.q.state_dict(),
            'optimizer_state_dict': self.q_opt.state_dict()}, path + '/q')

        torch.save({'model_state_dict': self.q_target.state_dict()}, path + '/q_target')

    def load_network(self, path):
        ''' Load network from path
        '''
        torch.load({'model_state_dict': self.q.state_dict(),
            'optimizer_state_dict': self.q_opt.state_dict()}, path + '/q')

        torch.load({'model_state_dict': self.q_target.state_dict()}, path + '/q_target')

    def _get_action_idx(self, state):
        ''' Return the index of the action in the action space
        '''
        # exploratory action
        if (np.random.rand() <= self.epsilon):
            return np.random.randint(self.action_dim)
        # greedy action
        self.q.eval()
        with torch.no_grad():
            act_idx = torch.argmin(self.q(state))
        self.q.train()
        return act_idx
