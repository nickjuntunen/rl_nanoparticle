import torch.distributions as ptd


class BasePolicy:
    def action_distribution(self, state):
        raise NotImplementedError
    
    def act(self, state, return_log_prob=False):
        distribution = self.action_distribution(state)
        sampled_actions = distribution.sample()
        log_probs = distribution.log_prob(sampled_actions)
        if return_log_prob:
            return sampled_actions, log_probs
        return sampled_actions
    

class ActorCriticPolicy(BasePolicy):
    def __init__(self, ac):
        self.ac = ac
        
    def action_distribution(self, state):
        action_mean, action_std, _ = self.ac(state)
        normal = ptd.Normal(action_mean, action_std)
        distribution = ptd.Independent(normal, 1)
        return distribution
