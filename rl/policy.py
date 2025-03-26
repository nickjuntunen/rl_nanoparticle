import torch.distributions as ptd
import torch


class BasePolicy:
    def action_distribution(self, state):
        raise NotImplementedError
    
    def act(self, state, return_log_prob=False):
        distribution = self.action_distribution(state)
        sampled_actions = distribution.sample()
        if return_log_prob:
            log_probs = distribution.log_prob(sampled_actions)
            return sampled_actions, log_probs
        return sampled_actions
    

class GaussianActorCriticPolicy(BasePolicy):
    def __init__(self, ac):
        self.ac = ac
        
    def action_distribution(self, state):
        action_mean, action_std, _ = self.ac(state)
        normal = ptd.Normal(action_mean, action_std)
        distribution = ptd.Independent(normal, 1)
        return distribution


class CategoricalActorCriticPolicy(BasePolicy):
    def __init__(self, ac, action_space):
        self.ac = ac
        self.action_space = action_space
        self.num_actions = len(action_space)
        
    def action_distribution(self, state):
        action_logits, _, _ = self.ac(state)
        action_logits = action_logits.view(-1, self.num_actions)
        distribution = ptd.Categorical(logits=action_logits)
        return distribution

    def act(self, state, return_log_prob=False):
        distribution = self.action_distribution(state)
        action_indices = distribution.sample().to(dtype=torch.long, device=state.device)
        sampled_actions = self.action_space[action_indices]
        if return_log_prob:
            log_probs = distribution.log_prob(action_indices)
            return sampled_actions, log_probs
        return sampled_actions