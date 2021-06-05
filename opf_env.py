
import gym
import numpy as np
import pandapower as pp

from penalties import (voltage_violation, line_overload, apparent_overpower)


class OpfEnv(gym.Env):
    def __init__(self, net, objective,
                 obs_keys, obs_space, act_keys, act_space, sample_keys=None,
                 u_penalty=20, overload_penalty=2,
                 apparent_power_penalty=5, single_step=True,
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

        return obs, reward, done, info

    def _apply_actions(self, action):
        """ Apply agent actions to the power system at hand. """
        counter = 0
        for unit_type, actuator, idxs in self.act_keys:
            a = action[counter:counter + len(idxs)]
            self.net[unit_type][actuator].loc[idxs] = a
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
        return penalty

    def _set_random_state(self):
        """ Standard pre - implemented method to set power system to a new random
        state from uniform sampling. Uses the observation space as basis.
        """
        for unit_type, actuator, idxs in self.sample_keys:
            low = self.net[unit_type][f'min_{actuator}'].loc[idxs]
            high = self.net[unit_type][f'max_{actuator}'].loc[idxs]
            r = np.random.uniform(low, high, size=(len(idxs),))
            self.net[unit_type][actuator].loc[idxs] = r

    def _get_obs(self):
        obss = [(self.net[unit_type][column].loc[idxs])
                for unit_type, column, idxs in self.obs_keys]
        return np.concatenate(obss)

    def reset(self):
        self._sampling()
        return self._get_obs()

    def render(self, mode='human'):
        pass  # TODO
