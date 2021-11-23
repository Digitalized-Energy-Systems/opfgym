
import abc
import random
import pdb

import gym
import numpy as np
import pandapower as pp

from .penalties import (
    voltage_violation, line_trafo_overload, apparent_overpower)


class OpfEnv(gym.Env, abc.ABC):
    def __init__(self,  # net,
                 # obs_keys, obs_space, act_keys, act_space, sample_keys=None,
                 u_penalty=300, overload_penalty=2, ext_overpower_penalty=100,
                 apparent_power_penalty=500, active_power_penalty=100,
                 single_step=True,  # sampling=None, bus_wise_obs=False  # TODO
                 ):

        self.observation_space = get_obs_space(self.net, self.obs_keys)

        self.u_penalty = u_penalty
        self.overload_penalty = overload_penalty
        self.apparent_power_penalty = apparent_power_penalty
        self.active_power_penalty = active_power_penalty
        self.ext_overpower_penalty = ext_overpower_penalty

        self.single_step = single_step  # TODO: Multi-step episodes not implemented yet

        # Full state of the system (available in training, but not in testing)
        self.state = None  # TODO: Not implemented yet

        self.test = False

        # Is a powerflow calculation required to get new observations in reset?
        self.res_for_obs = False
        for unit_type, _, _ in self.obs_keys:
            if 'res_' in unit_type:
                self.res_for_obs = True
                break

        self._sampling()
        self._run_pf()

    @abc.abstractmethod
    def _calc_reward(self, net):
        pass

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

        return obs, reward - info['penalty'], done, info

    def _apply_actions(self, action):
        """ Apply agent actions to the power system at hand. """
        counter = 0
        # ignore invalid actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for unit_type, actuator, idxs in self.act_keys:
            a = action[counter:counter + len(idxs)]
            # Actions are relative to the maximum possible (scaled) value
            # Attention: The negative range is always equal to the pos range!
            # TODO: maybe use action wrapper instead?!
            max_action = self.net[unit_type][f'max_max_{actuator}'].loc[idxs]
            try:
                new_values = (a * max_action /
                              self.net[unit_type].scaling.loc[idxs])
            except AttributeError:
                # Scaling sometimes not existing -> TODO: maybe catch this once in init
                new_values = a * max_action

            self.net[unit_type][actuator].loc[idxs] = new_values
            counter += len(idxs)

        assert counter == len(action)

    def _run_pf(self):
        pp.runpp(self.net, voltage_depend_loads=False)

    def _calc_penalty(self):
        """ Constraint violations result in a penalty that can be subtracted
        from the reward.
        Standard penalties: voltage band, overload of lines & transformers. """
        penalty = 0
        penalty += voltage_violation(self.net, self.u_penalty)
        penalty += line_trafo_overload(self.net, self.overload_penalty, 'line')
        penalty += line_trafo_overload(self.net,
                                       self.overload_penalty, 'trafo')
        return penalty

    def _sampling(self, sample_keys=None):
        """ Standard pre-implemented method to set power system to a new random
        state from uniform sampling. Uses the observation space as basis.
        Requirement: For every observations there must be "min_{obs}" and
        "max_{obs}" given as range to sample from.
        """
        if not sample_keys:
            sample_keys = self.obs_keys
        for unit_type, column, idxs in sample_keys:
            self._sample_from_range(unit_type, column, idxs)

    def _sample_from_range(self, unit_type, column, idxs):
        low = self.net[unit_type][f'min_{column}'].loc[idxs]
        high = self.net[unit_type][f'max_{column}'].loc[idxs]
        r = np.random.uniform(low, high, size=(len(idxs),))
        self.net[unit_type][column].loc[idxs] = r

    def _set_simbench_state(self, step: int=None):
        """ Standard pre-implemented method to sample a random state from the
        simbench time-series data and set that state.
        Works only for simbench systems!
        """
        noise_factor = 0.1
        if step is None:
            step = random.randint(0, len(self.profiles) - 1)
        # TODO: Consider some test steps that do not get sampled!
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
        self._sampling()
        if self.res_for_obs:
            self._run_pf()
        return self._get_obs()

    def render(self, mode='human'):
        pass  # TODO?

    def get_current_actions(self):
        # Scaling not considered here yet
        action = [(self.net[f'res_{unit_type}'][column].loc[idxs]
                   / self.net[unit_type][f'max_{column}'].loc[idxs])
                  for unit_type, column, idxs in self.act_keys]
        return np.concatenate(action)

    def test_step(self, action):
        """ TODO Use some custom data from different distribution here. For
        example some subset of the simbench data that is not used in training """
        obs, reward, done, info = self.step(action)

        # TODO: Automatically compare with OPF here?

        # Don't consider the penalty, to compare how good objective was learned
        print('Penalty: ', info['penalty'])
        return obs, reward + info['penalty'], done, info

    def baseline_reward(self):
        """ Compute some baseline to compare training performance with. In this
        case, use the optimal possible reward, which can be computed with the
        optimal power flow. """
        success = self._optimal_power_flow()
        if not success:
            return np.nan
        reward = self._calc_reward(self.net)
        penalty = self._calc_penalty()
        print('Penalty: ', penalty)

        return reward - penalty

    def _optimal_power_flow(self):
        try:
            # TODO: Make sure that this does not change the actual grid, but only a copy of it
            pp.runopp(self.net)
        except pp.optimal_powerflow.OPFNotConverged:
            print('OPF not converged!!!')
            return False
        return True


def get_obs_space(net, obs_keys: list):
    """ Get observation space from the constraints of the power network. """
    lows, highs = [], []
    for unit_type, column, idxs in obs_keys:
        if 'res_' in unit_type:
            # The constraints are never defined in the results table
            unit_type = unit_type[4:]
        lows.append(net[unit_type][f'min_{column}'].loc[idxs])
        highs.append(net[unit_type][f'max_{column}'].loc[idxs])

    return gym.spaces.Box(
        np.concatenate(lows, axis=0), np.concatenate(highs, axis=0))
