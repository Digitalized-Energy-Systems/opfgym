
import pdb

import gym
import numpy as np
import pandapower as pp

from .penalties import (
    voltage_violation, line_overload, apparent_overpower, active_overpower)


class OpfEnv(gym.Env):
    def __init__(self, net, objective,
                 obs_keys, obs_space, act_keys, act_space, sample_keys=None,
                 u_penalty=20, overload_penalty=2,
                 apparent_power_penalty=5, active_power_penalty=5,
                 single_step=True,
                 sampling=None, bus_wise_obs=False  # TODO
                 ):
        self.net = net
        self.observation_space = obs_space
        self.obs_keys = obs_keys
        self.action_space = act_space
        self.act_keys = act_keys
        if sample_keys:
            self.sample_keys = sample_keys
        else:
            self.sample_keys = obs_keys

        self._calc_reward = objective
        self.u_penalty = u_penalty
        self.overload_penalty = overload_penalty
        self.apparent_power_penalty = apparent_power_penalty
        self.active_power_penalty = active_power_penalty

        if not sampling:
            self._sampling = self._set_random_state
        else:
            self._sampling = sampling

        self.single_step = single_step

        # Full state of the system (available in training, but not in testing)
        self.state = None  # TODO

    def step(self, action):
        self._apply_actions(action)
        self._run_pf()
        reward = self._calc_reward(self.net)

        if self.single_step:
            done = True
        else:
            raise NotImplementedError

        obs = self._get_obs()
        info = {'penalty': self._calc_penalty()}

        print(action)
        print(obs)
        print(reward)
        print(info['penalty'])
        print('')

        return obs, reward - info['penalty'], done, info

    def _apply_actions(self, action):
        """ Apply agent actions to the power system at hand. """
        counter = 0
        for unit_type, actuator, idxs in self.act_keys:
            a = action[counter:counter + len(idxs)]
            # Actions are relative to the maximum possible value
            # Attention: If negative actions are possible, min=max!
            # TODO: maybe use action wrapper instead?!
            new_values = a * self.net[unit_type][f'max_{actuator}'].loc[idxs]
            self.net[unit_type][actuator].loc[idxs] = new_values
            counter += len(idxs)

        assert counter == len(action)

    def _run_pf(self):
        pp.runpp(self.net)

    def _calc_penalty(self):
        """ Constraint violations result in a penalty that can be subtracted
        from the reward. """
        penalty = 0
        penalty += voltage_violation(self.net, self.u_penalty)
        penalty += line_overload(self.net, self.overload_penalty)
        penalty += apparent_overpower(self.net, self.apparent_power_penalty)

        # TODO: Currently not useful, because power set-points are relative
        # anyway -> relative or absolute actions better
        # (p = action or p = action*p_max)???
        # penalty += active_overpower(self.net, self.apparent_power_penalty)

        return penalty

    def _ignore_invalid_actions(self):
        """ Prevent the agent from "cheating" by ignoring invalid actions, for
        example too high power values of the generators. """
        pass

    def _set_random_state(self):
        """ Standard pre-implemented method to set power system to a new random
        state from uniform sampling. Uses the observation space as basis.
        Requirement: For every observations there must be "min_{obs}" and
        "min_{obs}" given as range to sample from.
        """
        for unit_type, column, idxs in self.sample_keys:
            low = self.net[unit_type][f'min_{column}'].loc[idxs]
            high = self.net[unit_type][f'max_{column}'].loc[idxs]
            r = np.random.uniform(low, high, size=(len(idxs),))
            self.net[unit_type][column].loc[idxs] = r

    def _get_obs(self):
        obss = [(self.net[unit_type][column].loc[idxs])
                for unit_type, column, idxs in self.obs_keys]
        return np.concatenate(obss)

    def reset(self):
        self._sampling()
        return self._get_obs()

    def render(self, mode='human'):
        pass  # TODO

    def get_optimal_actions(self):
        # import pdb
        # pdb.set_trace()
        try:
            pp.runopp(self.net)
        except pp.optimal_powerflow.OPFNotConverged:
            print('OPF not converged')
            return None
        print('OPF converged!!!')

        return self._get_last_actions()

    def _get_last_actions(self):
        action = [(self.net[f'res_{unit_type}'][column].loc[idxs])
                  for unit_type, column, idxs in self.act_keys]
        return np.concatenate(action)
