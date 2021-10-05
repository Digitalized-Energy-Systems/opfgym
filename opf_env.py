
import pdb

import gym
import numpy as np
import pandapower as pp

from .penalties import (
    voltage_violation, line_trafo_overload, apparent_overpower,
    active_overpower, ext_grid_overpower)


class OpfEnv(gym.Env):
    def __init__(self, net, objective,
                 obs_keys, obs_space, act_keys, act_space, sample_keys=None,
                 u_penalty=300, overload_penalty=2, ext_overpower_penalty=100,
                 apparent_power_penalty=500, active_power_penalty=100,
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
        self.ext_overpower_penalty = ext_overpower_penalty

        if not sampling:
            self._sampling = self._set_random_state
        elif sampling == 'simbench':
            assert not sb.profiles_are_missing(net)
            self.profiles = sb.get_absolute_values(net,
                                                   profiles_instead_of_study_cases=True)
            self._sampling = self._set_simbench_state
        else:
            self._sampling = sampling

        self.single_step = single_step

        # Full state of the system (available in training, but not in testing)
        self.state = None  # TODO

        self.test = False

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

        # print('action:', action)
        # print('obs:', obs)
        # print(reward)
        # print(info['penalty'])
        # print('')

        return obs, reward - info['penalty'], done, info

    def _apply_actions(self, action):
        """ Apply agent actions to the power system at hand. """
        counter = 0
        # ignore invalid actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for unit_type, actuator, idxs in self.act_keys:
            a = action[counter:counter + len(idxs)]
            # Actions are relative to the maximum possible value
            # Attention: If negative actions are possible, min=max! (TODO)
            # TODO: maybe use action wrapper instead?!
            # TODO: Ensure that no invalid actions are used! (eg negative p)
            new_values = a * \
                self.net[unit_type][f'max_max_{actuator}'].loc[idxs]
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
        penalty += line_trafo_overload(self.net, self.overload_penalty, 'line')
        penalty += line_trafo_overload(self.net,
                                       self.overload_penalty, 'trafo')
        penalty += apparent_overpower(self.net, self.apparent_power_penalty)
        penalty += ext_grid_overpower(self.net,
                                      self.ext_overpower_penalty, 'q_mvar')

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
        "max_{obs}" given as range to sample from.
        """
        for unit_type, column, idxs in self.sample_keys:
            low = self.net[unit_type][f'min_{column}'].loc[idxs]
            high = self.net[unit_type][f'max_{column}'].loc[idxs]
            r = np.random.uniform(low, high, size=(len(idxs),))
            self.net[unit_type][column].loc[idxs] = r

    def _set_simbench_state(self):
        """ Standard pre-implemented method to sample a random state from the
        simbench time-series data and set that state.
        Works only for simbench systems!
        """
        noise_factor = 0.1
        step = random.randint(0, len(self.profiles) - 1)
        for type_act in self.profiles.keys():
            if not self.profiles[type_act].shape[1]:
                continue
            unit_type, actuator = type_act

            # Add some noise to create unique data samples
            noise = np.random.random(
                len(self.net[unit_type].index)) * noise_factor * 2 + (1 - noise_factor)
            new_values = self.profiles[type_act].loc[step] * noise
            self.net[unit_type].loc[:, actuator] = new_values

            # Make sure no boundaries are violated for generators
            if unit_type == 'sgen':
                self.net.sgen.loc[:, actuator] = self.net.sgen[
                    [actuator, f'max_{actuator}']].min(axis=1)

    def _get_obs(self):
        obss = [(self.net[unit_type][column].loc[idxs])
                for unit_type, column, idxs in self.obs_keys]
        return np.concatenate(obss)

    def reset(self):
        self._sampling(self)
        return self._get_obs()

    def render(self, mode='human'):
        pass  # TODO

    def _optimal_power_flow(self):
        try:
            # TODO: Make sure that this does not change the actual grid, but only a copy of it
            pp.runopp(self.net)
        except pp.optimal_powerflow.OPFNotConverged:
            print('OPF not converged!!!')
            return False
        return True

    def get_current_actions(self):
        action = [(self.net[f'res_{unit_type}'][column].loc[idxs]
                   / self.net[unit_type][f'max_{column}'].loc[idxs])
                  for unit_type, column, idxs in self.act_keys]
        return np.concatenate(action)

    def test_step(self, action):
        """ TODO Use some custom data from different distribution here. """
        result = self.step(action)

        # print('action: ', action)
        # success = self._optimal_power_flow()
        # if not success:
        #     print('failure')
        #     return result
        # # print('optimal action:', self.get_current_actions())
        # print('reward: ', result[1])
        # opt_reward = self._calc_reward(self.net) - self._calc_penalty()
        # print('opt reward: ', opt_reward)
        # print('MSE: ', 100 * abs((opt_reward - result[1]) / opt_reward))

        return result

    def baseline_reward(self):
        """ Compute some baseline to compare training performance with. In this
        case, use the optimal possible reward, which can be computed with the
        optimal power flow. """
        success = self._optimal_power_flow()
        if not success:
            return np.nan
        reward = self._calc_reward(self.net) - self._calc_penalty()
        print('penalty: ', self._calc_penalty())
        return reward
