
import abc
import random
import warnings

import gym
import numpy as np
import pandapower as pp

from .penalties import (voltage_violation, line_trafo_overload)

warnings.simplefilter('once')


# TODO: Calc reward from pandapower cost function (for OPF comparison)


# Use one week every two months as test data
one_week = 7 * 24 * 4
TEST_DATA = np.append(
    np.arange(one_week),
    [np.arange(9 * one_week, 10 * one_week),
     np.arange(18 * one_week, 19 * one_week),
     np.arange(27 * one_week, 28 * one_week),
     np.arange(36 * one_week, 37 * one_week),
     np.arange(45 * one_week, 46 * one_week)]
)


class OpfEnv(gym.Env, abc.ABC):
    def __init__(self, u_penalty=300, overload_penalty=2, ext_overpower_penalty=100,
                 apparent_power_penalty=500, active_power_penalty=100,
                 train_test_split=False,
                 vector_reward=False, single_step=True,
                 # TODO: Idea to put obs together bus-wise instead of unit-wise
                 bus_wise_obs=False,
                 # Idea: add voltages and loadings to obs
                 full_obs=False,
                 autocorrect_prio='p_mw',
                 pf_for_obs=None, use_time_obs=False, seed=None):

        self.train_test_split = train_test_split

        self.use_time_obs = use_time_obs

        if full_obs:
            self.obs_keys.extend([
                ('res_bus', 'vm_pu', self.net.bus.index),
                ('res_line', 'loading_percent', self.net.line.index),
                ('res_trafo', 'loading_percent', self.net.trafo.index)])

        self.observation_space = get_obs_space(
            self.net, self.obs_keys, use_time_obs, seed)

        self.vector_reward = vector_reward

        if vector_reward is True:
            # 3 penalties and one objective function
            self.reward_space = gym.spaces.Box(
                low=-np.ones(4) * np.inf, high=np.ones(4) * np.inf, seed=seed)

        self.u_penalty = u_penalty
        self.overload_penalty = overload_penalty
        self.apparent_power_penalty = apparent_power_penalty
        self.active_power_penalty = active_power_penalty
        self.ext_overpower_penalty = ext_overpower_penalty

        self.priority = autocorrect_prio

        self.single_step = single_step  # TODO: Multi-step episodes not implemented yet

        # Full state of the system (available in training, but not in testing)
        self.state = None  # TODO: Not implemented yet
        self.test = False

        # Is a powerflow calculation required to get new observations in reset?
        self.pf_for_obs = pf_for_obs
        if pf_for_obs is None:
            # Automatic checking
            for unit_type, _, _ in self.obs_keys:
                if 'res_' in unit_type:
                    self.pf_for_obs = True
                    break

    @abc.abstractmethod
    def _calc_reward(self, net):
        pass

    def reset(self, step=None):
        self._sampling(step)
        # Reset all actions to default values
        default_act = (self.action_space.low + self.action_space.high) / 2
        self._apply_actions(default_act)

        if self.pf_for_obs is True:
            success = self._run_pf()
            if not success:
                print('Failed powerflow calculcation in reset. Try again!')
                return self.reset()
        return self._get_obs(self.obs_keys, self.use_time_obs)

    def step(self, action):
        assert not np.isnan(action).any()
        info = {}

        self._apply_actions(action)
        self._autocorrect_apparent_power(self.priority)
        success = self._run_pf()

        if not success:
            # Something went seriously wrong! Find out what!
            # Maybe NAN in power setpoints?!
            # Maybe simply catch this with a strong negative reward?!
            import pdb
            pdb.set_trace()

        reward = self._calc_reward(self.net)
        info['penalty'] = self._calc_penalty()

        if self.single_step:
            done = True
        elif random.random() < 0.02:  # TODO! Better termination criterion
            self._sampling(step=self.current_step + 1)
            done = True  # TODO
            info['TimeLimit.truncated'] = True
        else:
            done = not self._sampling(step=self.current_step + 1)
            info['TimeLimit.truncated'] = True

        obs = self._get_obs(self.obs_keys, self.use_time_obs)
        if np.isnan(obs).any():
            import pdb
            pdb.set_trace()
        assert not np.isnan(obs).any()

        if not self.vector_reward:
            reward += sum(info['penalty'])
        else:
            # Reward as a vector
            reward = np.append(reward, info['penalty'])

        return obs, reward, done, info

    def _apply_actions(self, action, autocorrect=False):
        """ Apply agent actions to the power system at hand. """
        counter = 0
        # Clip invalid actions
        action = np.clip(action,
                         self.action_space.low[:len(action)],
                         self.action_space.high[:len(action)])
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

    def _autocorrect_apparent_power(self, priority='p_mw'):
        """ Autocorrect to maximum apparent power if necessary. Relevant for
        sgens, loads, and storages """
        not_prio = 'p_mw' if priority == 'q_mvar' else 'q_mvar'
        for unit_type in ('sgen', 'load', 'storage'):
            df = self.net[unit_type]
            if 'max_s_mva' in df.columns:
                s_mva2 = df.max_s_mva.to_numpy() ** 2
                values2 = (df[priority] * df.scaling).to_numpy() ** 2
                # Make sure to prevent negative values for sqare root
                max_values = np.maximum(s_mva2 - values2, 0)**0.5 / df.scaling
                # Reduce non-priority power column
                self.net[unit_type][not_prio] = np.sign(df[not_prio]) * \
                    np.minimum(df[not_prio].abs(), max_values)

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

    def _sampling(self, *args, **kwargs):
        self._set_simbench_state(*args, **kwargs)

    def _sample_uniform(self, sample_keys=None):
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

    def _set_simbench_state(self, step: int=None, test=False,
                            noise_factor=0.1, noise_distribution='uniform'):
        """ Standard pre-implemented method to sample a random state from the
        simbench time-series data and set that state.
        Works only for simbench systems!
        """

        # Use one week every two months as test data (12%)
        # TODO

        if step is None:
            total_n_steps = len(self.profiles[('load', 'q_mvar')])
            while True:
                step = random.randint(0, total_n_steps - 1)
                if test is False:
                    if self.train_test_split and (
                            step in TEST_DATA or step + 20 in TEST_DATA):
                        # Do not sample too close to test data range
                        continue
                    break
                elif test is True:
                    # TODO
                    raise NotImplementedError

        if self.train_test_split and step in TEST_DATA:
            # Next step would be test data -> end of episode
            return False

        if step > len(self.profiles[('load', 'q_mvar')]) - 1:
            # End of time series data
            return False

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

        return True

    def _get_obs(self, obs_keys, use_time_obs):
        obss = [(self.net[unit_type][column].loc[idxs].to_numpy())
                for unit_type, column, idxs in obs_keys]

        if use_time_obs:
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
                   / self.net[unit_type][f'max_max_{column}'].loc[idxs])
                  for unit_type, column, idxs in self.act_keys]
        return np.concatenate(action)

    # def test_step(self, action):
    #     """ TODO Use some custom data from different distribution here. For
    #     example some subset of the simbench data that is not used in training """
    #     obs, reward, done, info = self.step(action)

    #     # TODO: Automatically compare with OPF here?

    #     print('Test Penalty: ', info['penalty'])
    #     print('Current actions: ', self.get_current_actions())
    #     return obs, reward, done, info

    #     # Don't consider the penalty, to compare how good objective was learned?
    #     # if self.vector_reward:
    #     #     # Only return objective reward, not penalty reward
    #     #     return obs, reward[0], done, info
    #     # else:
    #     #     # Remove previously added penalty
    #     #     return obs, reward - sum(info['penalty']), done, info

    def compute_error(self, action):
        """ Return error compared to optimal state of the system. """

        # Perform (non-optimal) action
        self._apply_actions(action)
        self._autocorrect_apparent_power(self.priority)
        success = self._run_pf()

        if not success:
            return np.nan, np.nan

        reward = self._calc_reward(self.net)
        penalty = self._calc_penalty()

        obj = sum(np.append(reward, penalty))

        print('Test Penalty: ', penalty)
        print('Current actions: ', self.get_current_actions())

        opt_obj = self.baseline_reward()

        return opt_obj, obj

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
        print('Baseline actions: ', self.get_current_actions())

        return sum(np.append(reward, penalty))

    def _optimal_power_flow(self):
        try:
            # TODO: Make sure that this does not change the actual grid, but only a copy of it
            pp.runopp(self.net)
        except pp.optimal_powerflow.OPFNotConverged:
            print('OPF not converged!!!')
            return False
        return True


def get_obs_space(net, obs_keys: list, use_time_obs: bool, seed: int,
                  last_n_obs: int=1):
    """ Get observation space from the constraints of the power network. """
    lows, highs = [], []

    if use_time_obs:
        # Time is always given as observation of lenght 6 in range [-1, 1]
        # at the beginning of the observation!
        lows.append(-np.ones(6))
        highs.append(np.ones(6))

    for unit_type, column, idxs in obs_keys:
        if 'res_' in unit_type:
            # The constraints are never defined in the results table
            unit_type = unit_type[4:]

        try:
            l = net[unit_type][f'min_{column}'].loc[idxs].to_numpy()
            h = net[unit_type][f'max_{column}'].loc[idxs].to_numpy()
        except KeyError:
            # Special case: trafos and lines (have minimum constraint of zero)
            l = np.zeros(len(idxs))
            h = net[unit_type][f'max_{column}'].loc[idxs].to_numpy() * 1.5

        # Special case: voltages
        if column == 'vm_pu' or unit_type == 'ext_grid':
            diff = h - l
            l = l - diff / 2
            h = h + diff / 2

        try:
            if 'min' in column or 'max' in column:
                # Constraints need to remain scaled
                raise AttributeError
            for _ in range(last_n_obs):
                lows.append(l / net[unit_type].scaling.loc[idxs].to_numpy())
                highs.append(h / net[unit_type].scaling.loc[idxs].to_numpy())
        except AttributeError:
            print(f'Scaling for {unit_type} not defined: assume scaling=1')
            for _ in range(last_n_obs):
                lows.append(l)
                highs.append(h)

    return gym.spaces.Box(
        np.concatenate(lows, axis=0), np.concatenate(highs, axis=0), seed=seed)
