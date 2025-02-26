import torch
import numpy as np
import kmc_lattice_gas as kmc


class EnvArgs:
    def __init__(self):
        self.seed=10
        self.enn=7.0
        self.ens=7.0
        # self.lower=0.5
        # self.delay=50.0
        self.time_dependent_rates=False
        self.save_lattice_traj=True
        self.enf=1.0
        self.esf=1.0
        self.eff=1.0
        self.muf=1.0
        self.mun=5.0
        self.fug=0.0
        self.temperature=0.1
        self.diffusion_prefactor=1.0
        self.fluctuation_prefactor=1.0
        self.rotational_prefactor=10.0
        self.height=10
        self.n_side=50
        # self.amplitude=0.5
        # self.frequency=5
        self.update_type = 'global_ens'
        self.epsilon = 0.1
        self.target_dist = None
    def get_list(self):
        return [self.seed, self.enn, self.ens, self.lower, self.delay, self.time_dependent_rates, self.save_lattice_traj, self.enf, self.esf, self.eff, self.muf, self.mun, self.fug, self.temperature, self.diffusion_prefactor, self.fluctuation_prefactor, self.rotational_prefactor, self.height, self.n_side, self.amplitude, self.frequency]


class Env:
    def __init__(self):
        pass

    def _get_reward(self):
        raise NotImplementedError

    def take_action(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    

class KMCEnv(Env):
    def __init__(self, args : EnvArgs):
        super(KMCEnv, self).__init__()
        args_list = args.get_list()
        init_args = args_list[:-4]
        self.sim = kmc.Simulation(*init_args)
        self.args = args
        self.state = self._get_state()[1]
        self.time = self.sim.time
        self.update_type = args.update_type
        self.epsilon = args.epsilon
        self.target_dist = args.target_dist
        self.area = args.n_side ** 2
        self.num_np = 0
        self.kl = torch.nn.KLDivLoss(reduction='sum')

    def reset(self):
        self.sim.reset()
        return self._get_state()[1]

    def _take_action(self, action):
        if self.update_type == 'temp':
            self.sim.take_action(action, True)
        elif self.update_type == 'global_ens':
            self.sim.take_action(action, False)
        elif self.update_type == 'local_ens':
            self.sim.take_action(action)

    def step(self, n_steps, act_idx, actions):
        self._take_action(actions[act_idx])
        self.sim.step(n_steps)
        self.time = self.sim.time
        self.num_np = self.sim.num_np
        box, state = self._get_state()
        reward = self._get_reward(state)
        return state, reward

    def _get_state(self):
        ''' Return the sim_box, cluster distribution, and number of particles in the system
        '''
        # sim_box
        sim_box = self.sim.get_box()
        sim_box = np.reshape(sim_box, (self.args.height, self.args.n_side, self.args.n_side))
        sim_box = torch.from_numpy(sim_box).float().permute(2,1,0)
        # cluster distribution
        cluster_array = self.sim.get_state()
        cluster_array = torch.from_numpy(cluster_array).float()
        # idx: cluster size, value at idx: number of clusters of that size
        return sim_box, cluster_array

    def _get_reward(self, state):
        ''' Return the reward
        '''
        kl_term = -self.kl(torch.log(state), torch.log(self.target_dist))
        growth_term = self.sim.num_np / self.area
        time_term = -self.time * 0.0001
        return kl_term + growth_term + time_term

    def get_state_reward(self):
        state = self._get_state()[1]
        reward = self._get_reward(state)
        return state, reward


    def print_state(self):
        self.sim.print_state()