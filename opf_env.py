
import abc
import random
import pdb
import warnings

import gym
import numpy as np
import pandapower as pp

from .penalties import (
    voltage_violation, line_trafo_overload, apparent_overpower)

warnings.simplefilter('once')


# TODO: Calc reward from pandapower cost function (for OPF comparison)

class OpfEnv(gym.Env, abc.ABC):
    def __init__(self, u_penalty=300, overload_penalty=2, ext_overpower_penalty=100,
                 apparent_power_penalty=500, active_power_penalty=100,
                 vector_reward=False, single_step=True, bus_wise_obs=False,  # TODO
                 use_time_obs=True):

        self.use_time_obs = use_time_obs
        self.observation_space = get_obs_space(
            self.net, self.obs_keys, use_time_obs)

        self.vector_reward = vector_reward
        if vector_reward is True:
            # 3 penalties and one objective function
            self.reward_space = gym.spaces.Box(
                low=-np.ones(4) * np.inf, high=np.ones(4) * np.inf)

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

    @abc.abstractmethod
    def _calc_reward(self, net):
        pass

    def reset(self, step=None):
        self._sampling(step)
        if self.res_for_obs is True:
            success = self._run_pf()
            if not success:
                print('Failed powerflow calculcation in reset. Try again!')
                return self.reset()
        return self._get_obs()

    def step(self, action):
        assert not np.isnan(action).any()
        self._apply_actions(action)
        success = self._run_pf()
        reward = self._calc_reward(self.net)

        if self.single_step:
            done = True
        else:
            raise NotImplementedError

        obs = self._get_obs()
        assert not np.isnan(obs).any()

        info = {'penalty': self._calc_penalty()}

        if not self.vector_reward:
            reward += sum(info['penalty'])
        else:
            # Reward as a vector
            reward = np.array([reward] + info['penalty'])
        return obs, reward, done, info

    def _apply_actions(self, action):
        """ Apply agent actions to the power system at hand. """
        counter = 0
        # ignore invalid actions
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for unit_type, actuator, idxs in self.act_keys:
            df = self.net[unit_type]
            a = action[counter:counter + len(idxs)]
            # Actions are relative to the maximum possible (scaled) value
            # Attention: The negative range is always equal to the pos range!
            # TODO: maybe use action wrapper instead?!
            max_action = df[f'max_max_{actuator}'].loc[idxs]
            new_values = a * max_action
            # Autocorrect impossible setpoints (however: no penalties this way)
            if f'max_{actuator}' in df.columns:
                mask = new_values > df[f'max_{actuator}'].loc[idxs]
                new_values[mask] = df[f'max_{actuator}'].loc[idxs][mask]
            if f'min_{actuator}' in df.columns:
                mask = new_values < df[f'min_{actuator}'].loc[idxs]
                new_values[mask] = df[f'min_{actuator}'].loc[idxs][mask]

            if 'scaling' in df.columns:
                new_values /= df.scaling.loc[idxs]
            # Scaling sometimes not existing -> TODO: maybe catch this once in init

            self.net[unit_type][actuator].loc[idxs] = new_values
            counter += len(idxs)

        if counter != len(action):
            warnings.warn('More actions than action keys!')

    def _run_pf(self):
        try:
            pp.runpp(self.net,
                     voltage_depend_loads=False,
                     enforce_q_lims=True)
        except pp.powerflow.LoadflowNotConverged:
            print('Powerflow not converged!!!')
            return False
        return True

    def _calc_penalty(self):
        """ Constraint violations result in a penalty that can be subtracted
        from the reward.
        Standard penalties: voltage band, overload of lines & transformers. """
        penalty = []
        penalty.append(-voltage_violation(self.net, self.u_penalty))
        penalty.append(-line_trafo_overload(
            self.net, self.overload_penalty, 'line'))
        penalty.append(-line_trafo_overload(
            self.net, self.overload_penalty, 'trafo'))
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

    def _set_simbench_state(self, step: int=None, noise_factor=0.1,
                            noise_distribution='uniform'):
        """ Standard pre-implemented method to sample a random state from the
        simbench time-series data and set that state.
        Works only for simbench systems!
        """
        if step is None:
            total_n_steps = len(self.profiles[('load', 'q_mvar')])
            step = random.randint(0, total_n_steps - 1)

        self.current_step = step
        # TODO: Consider some test steps that do not get sampled!
        for type_act in self.profiles.keys():
            if not self.profiles[type_act].shape[1]:
                continue
            unit_type, actuator = type_act

            data = self.profiles[type_act].loc[step, self.net[unit_type].index]
            # Add some noise to create unique data samples
            if noise_distribution == 'uniform':
                # Uniform distribution: noise_factor as relative sample range
                noise = np.random.random(
                    len(self.net[unit_type].index)) * noise_factor * 2 + (1 - noise_factor)
                new_values = data * noise
            elif noise_distribution == 'normal':
                # Normal distribution: noise_factor as relative std deviation
                new_values = np.random.normal(
                    loc=data, scale=data * noise_factor)

            # Make sure that the range of original data remains unchanged
            # (Technical limits of the units remain the same)
            new_values = np.clip(new_values,
                                 self.profiles[type_act].min(),
                                 self.profiles[type_act].max())

            self.net[unit_type].loc[:, actuator] = new_values

    def _get_obs(self):
        obss = [(self.net[unit_type][column].loc[idxs].to_numpy())
                for unit_type, column, idxs in self.obs_keys]

        if self.use_time_obs:
            obss = [self._get_time_observation()] + obss
        return np.concatenate(obss)

    def _get_time_observation(self):
        """ Return current time in sinus/cosinus form.
        Example daytime: (0.0, 1.0) = 00:00 and (1.0, 0.0) = 06:00. Idea from
        https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
        """
        total_n_steps = len(self.profiles[('load', 'q_mvar')])
        # number of steps per timeframe
        dayly, weekly, yearly = (24 * 4, 7 * 24 * 4, total_n_steps)
        time_obs = []
        for timeframe in (dayly, weekly, yearly):
            timestep = self.current_step % timeframe
            cyclical_time = 2 * np.pi * timestep / timeframe
            time_obs.append(np.sin(cyclical_time))
            time_obs.append(np.cos(cyclical_time))

        return np.array(time_obs)

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
        print('Test Penalty: ', info['penalty'])
        if self.vector_reward:
            # Only return objective reward, not penalty reward
            return obs, reward[0], done, info
        else:
            # Remove previously added penalty
            return obs, reward - sum(info['penalty']), done, info

    def baseline_reward(self):
        """ Compute some baseline to compare training performance with. In this
        case, use the optimal possible reward, which can be computed with the
        optimal power flow. """
        success = self._optimal_power_flow()
        if not success:
            return np.nan
        reward = self._calc_reward(self.net)
        penalty = self._calc_penalty()
        print('Base Penalty: ', penalty)

        return reward

    def _optimal_power_flow(self):
        try:
            # TODO: Make sure that this does not change the actual grid, but only a copy of it
            pp.runopp(self.net)
        except pp.optimal_powerflow.OPFNotConverged:
            print('OPF not converged!!!')
            return False
        return True


def get_obs_space(net, obs_keys: list, use_time_obs: bool):
    """ Get observation space from the constraints of the power network. """
    lows, highs = [], []

    if use_time_obs:
        # Time is always given as observation of lenght 6 in range [-1, 1]
        lows.append(np.array([-1] * 6))
        highs.append(np.array([1] * 6))

    for unit_type, column, idxs in obs_keys:
        if 'res_' in unit_type:
            # The constraints are never defined in the results table
            unit_type = unit_type[4:]
        try:
            if 'min' in column or 'max' in column:
                # Constraints need to remain scaled
                raise AttributeError
            lows.append((net[unit_type][f'min_{column}'].loc[idxs]
                         / net[unit_type].scaling.loc[idxs]).to_numpy())
            highs.append((net[unit_type][f'max_{column}'].loc[idxs]
                          / net[unit_type].scaling.loc[idxs]).to_numpy())
        except AttributeError:
            print(f'Scaling for {unit_type} not defined: assume scaling=1')
            lows.append(net[unit_type][f'min_{column}'].loc[idxs].to_numpy())
            highs.append(net[unit_type][f'max_{column}'].loc[idxs].to_numpy())

    return gym.spaces.Box(
        np.concatenate(lows, axis=0), np.concatenate(highs, axis=0))
