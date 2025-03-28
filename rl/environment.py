import torch
import numpy as np
import kmc_lattice_gas as kmc


class EnvArgs:
    def __init__(self):
        self.seed = 10
        self.enn = 7.0
        self.ens = 7.0
        # self.lower=0.5
        # self.delay=50.0
        # self.time_dependent_rates = False
        # self.save_lattice_traj = False
        self.enf = 1.0
        self.esf = 1.0
        self.eff = 1.0
        self.muf = 1.0
        self.mun = 5.0
        self.fug = 0.0
        self.temperature = 0.1
        self.diffusion_prefactor = 1.0
        self.fluctuation_prefactor = 1.0
        self.rotational_prefactor = 10.0
        self.height = 10
        self.n_side = 50
        # self.amplitude=0.5
        # self.frequency=5
        self.update_type = "global_ens"
        self.epsilon = 0.1
        self.target_dist = None

    def get_list(self):
        return [
            self.seed,
            self.enn,
            self.ens,
            # self.lower,
            # self.delay,
            # self.time_dependent_rates,
            # self.save_lattice_traj = False,
            self.enf,
            self.esf,
            self.eff,
            self.muf,
            self.mun,
            self.fug,
            self.temperature,
            self.diffusion_prefactor,
            self.fluctuation_prefactor,
            self.rotational_prefactor,
            self.height,
            self.n_side,
            # self.amplitude,
            # self.frequency,
        ]


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
    def __init__(self, args: EnvArgs):
        super(KMCEnv, self).__init__()
        args_list = args.get_list()
        self.sim = kmc.Simulation(*args_list)
        self.args = args
        self.sim_box, self.state = self._get_state()
        self.time = self.sim.time
        self.surface_coverage = self.sim.get_surface_coverage()
        self.update_type = args.update_type
        self.epsilon = args.epsilon

        if isinstance(args.target_dist, tuple):
            self.target_nclu = args.target_dist[0]
            self.target_area = args.target_dist[1]
        else:
            self.target = args.target_dist
        
        self.area = args.n_side**2
        self.num_np = 0
        self.ens = self.sim.ens_grid
        self.kl = torch.nn.KLDivLoss(reduction="sum")

    def reset(self):
        self.sim.reset()
        self.time = self.sim.time
        self.num_np = self.sim.num_np
        return self._get_state()[1]

    def _take_action(self, action):
        if self.update_type == "temp":
            self.sim.take_action(action, True)
        elif self.update_type == "global_ens":
            self.sim.take_action(action, False)
        elif self.update_type == "local_ens":
            action_list = []
            action = action.squeeze().flatten()
            for i in range(len(action)):
                action_list.append(action[i].item())
            self.sim.take_action(action_list)

    def step(self, n_steps, action, max_num_np):
        self._take_action(action)
        self.sim.step(n_steps)
        self.time = self.sim.time
        self.num_np = self.sim.num_np
        self.ens = self.sim.ens_grid
        self.surface_coverage = self.sim.get_surface_coverage()
        box, state = self._get_state()
        if self.update_type == "local_ens":
            box = box[:, :, 1]
            reward = self._get_reward(box, action)
            return box, reward
        reward = self._get_reward(state, max_num_np)
        return state, reward

    def _get_state(self):
        """Return the sim_box, cluster distribution, and number of particles in the system"""
        # sim_box
        sim_box = self.sim.get_box()
        sim_box = np.reshape(
            sim_box, (self.args.height, self.args.n_side, self.args.n_side)
        )
        sim_box = torch.from_numpy(sim_box).float().permute(2, 1, 0)
        # cluster distribution
        cluster_array = self.sim.get_state()
        cluster_array = torch.from_numpy(cluster_array).float()
        # idx: cluster size, value at idx: number of clusters of that size
        return sim_box, cluster_array

    def _get_reward(self, state, actions, max_num_np=None):
        """Return the reward"""
        if not self.update_type == "local_ens":
            nclu_diff = torch.abs(torch.sum(state) - self.target_nclu)
            nclu_reward = 1 - torch.clamp(nclu_diff / self.target_nclu, -1, 1)

            area_diff = torch.abs(self.surface_coverage - self.target_area)
            area_reward = 1 - torch.clamp(area_diff / self.target_area, -1, 1)

            num_np_reward = self.num_np / max_num_np if max_num_np else 0
            
            time_penalty = -min(self.time / 1e6, 1)
            
            reward = 0.35 * nclu_reward + 0.35 * area_reward + 0.2 * num_np_reward + 0.1 * time_penalty
            return torch.clamp(reward, -1.0, 1.0)
        
        else:
            state = state - 1 # fluid is 0, np is 1 for comparison
            neg_space_error = (self.target - state > 0).sum() / self.area
            pos_space_error = (self.target - state < 0).sum() / self.area
            error = neg_space_error + 2 * pos_space_error
            reward = 1 - error
            reward = torch.clamp(reward, -1.0, 1.0)
            return reward

    def get_state_reward(self, actions=None):
        state = self._get_state()[1]
        reward = self._get_reward(state, actions)
        return state, reward

    def print_state(self):
        self.sim.print_state()
