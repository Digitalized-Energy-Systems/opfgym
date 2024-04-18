
import abc
import copy
import logging
import random
import warnings

import gymnasium as gym
import numpy as np
import pandapower as pp
import pandas as pd
import scipy
from scipy import stats

from mlopf.penalties import (voltage_violation, line_overload,
                             trafo_overload, ext_grid_overpower)
from mlopf.objectives import min_pp_costs
from mlopf.util.normalization import get_normalization_params

warnings.simplefilter('once')


class OpfEnv(gym.Env, abc.ABC):
    def __init__(self,
                 train_test_split=True,
                 test_share=0.2,
                 vector_reward=False,
                 steps_per_episode=1,
                 autocorrect_prio='p_mw',
                 pf_for_obs=None,
                 bus_wise_obs=False,
                 diff_reward=False,
                 reward_function: str='summation',
                 reward_scaling: str=None,
                 reward_scaling_params: dict=None,
                 squash_reward=False,
                 remove_normal_obs=False,
                 add_res_obs=False,
                 add_time_obs=False,
                 add_act_obs=False,
                 train_data='simbench',
                 test_data='simbench',
                 sampling_kwargs: dict=None,
                 volt_pen_kwargs: dict=None,
                 line_pen_kwargs: dict=None,
                 trafo_pen_kwargs: dict=None,
                 ext_grid_pen_kwargs: dict=None,
                 autoscale_penalty=False,
                 autoscale_violations=True,
                 min_penalty=None,
                 penalty_weight=0.5,
                 penalty_obs_range: tuple=None,
                 test_penalty=None,
                 seed=None,
                 *args, **kwargs):

        # Should be always True. Maybe only allow False for paper investigation
        self.train_test_split = train_test_split
        self.train_data = train_data
        self.test_data = test_data
        self.sampling_kwargs = sampling_kwargs if sampling_kwargs else {}

        # Define the observation space
        if remove_normal_obs:
            # Completely overwrite the observation definition
            assert add_res_obs or add_time_obs or add_act_obs
            # Make sure to only remove redundant data and not e.g. price data
            remove_idxs = []
            for idx, (unit_type, column, _) in enumerate(self.obs_keys):
                if unit_type in ('load', 'sgen', 'gen') and column in ('p_mw', 'q_mvar'):
                    remove_idxs.append(idx)
            self.obs_keys = [value for index, value in enumerate(self.obs_keys)
                             if index not in remove_idxs]

        self.add_act_obs = add_act_obs
        if add_act_obs:
            # The agent can observe its previous actions
            self.obs_keys.extend(self.act_keys)
            # Does not make sense without observing results from previous act
            add_res_obs = True

        self.add_time_obs = add_time_obs
        # Add observations that require previous pf calculation
        if add_res_obs:
            self.obs_keys.extend([
                ('res_bus', 'vm_pu', self.net.bus.index),
                ('res_line', 'loading_percent', self.net.line.index),
                ('res_trafo', 'loading_percent', self.net.trafo.index),
                ('res_ext_grid', 'p_mw', self.net.ext_grid.index),
                ('res_ext_grid', 'q_mvar', self.net.ext_grid.index)
            ])

        if penalty_obs_range:
            n_penalties = 4 # TODO
            self.penalty_obs_space = gym.spaces.Box(
                low=np.ones(n_penalties) * penalty_obs_range[0], 
                high=np.ones(n_penalties) * penalty_obs_range[1], 
                seed=seed)
            self.test_penalty = test_penalty
        else:
            self.penalty_obs_space = None

        self.bus_wise_obs = bus_wise_obs
        self.observation_space = get_obs_space(
            self.net, self.obs_keys, add_time_obs, self.penalty_obs_space, seed,
            bus_wise_obs=bus_wise_obs)
        self.action_space = get_action_space(self.act_keys, seed)

        self.reward_function = reward_function
        self.vector_reward = vector_reward
        self.squash_reward = squash_reward

        self.volt_pen = volt_pen_kwargs if volt_pen_kwargs else {}
        self.line_pen = line_pen_kwargs if line_pen_kwargs else {}
        self.trafo_pen = trafo_pen_kwargs if trafo_pen_kwargs else {}
        self.ext_grid_pen = ext_grid_pen_kwargs if ext_grid_pen_kwargs else {}

        # self.autoscale_penalty = autoscale_penalty
        self.min_penalty = min_penalty
        self.autoscale_violations = autoscale_violations
        
        self.priority = autocorrect_prio

        self.steps_per_episode = steps_per_episode

        # Full state of the system (available in training, but not in testing)
        self.state = None  # TODO: Not implemented yet. Required only for partially observable envs

        # Is a powerflow calculation required to get new observations in reset?
        self.pf_for_obs = pf_for_obs
        if pf_for_obs is None:
            # Automatic checking
            for unit_type, _, _ in self.obs_keys:
                if 'res_' in unit_type:
                    self.pf_for_obs = True
                    break

        self.diff_reward = diff_reward
        if diff_reward:
            self.pf_for_obs = True

        self.test_steps = define_test_steps(test_share)

        # Prepare reward scaling for later on 
        self.reward_scaling = reward_scaling
        self.penalty_weight = penalty_weight
        reward_scaling_params = reward_scaling_params if reward_scaling_params else {}
        if reward_scaling_params == 'auto' or (
                not reward_scaling_params and reward_scaling):
            # Find reward range by trial and error
            params = get_normalization_params(self, num_samples=1000)
        else:
            params = reward_scaling_params

        if not reward_scaling:
            self.reward_factor = 1
            self.reward_bias = 0
            self.penalty_factor = 1
            self.penalty_bias = 0
        elif reward_scaling == 'minmax':
            # Scale from range [min, max] to range [-1, 1]
            # formula: (obj - min_obj) / (max_obj - min_obj) * 2 - 1
            diff = (params['max_obj'] - params['min_obj']) / 2
            self.reward_factor = 1 / diff
            self.reward_bias = -(params['min_obj'] / diff + 1)
            diff = 2 * (params['max_viol'] - params['min_viol'])
            self.penalty_factor = 1 / diff
            self.penalty_bias = params['min_viol'] / diff + 1
        elif reward_scaling == 'normalization':
            # Scale so that mean is zero and standard deviation is one
            # formula: (obj - mean_obj) / obj_std
            self.reward_factor = 1 / params['std_obj']
            self.reward_bias = -params['mean_obj'] / params['std_obj']
            self.penalty_factor = 1 / params['std_viol']
            self.penalty_bias = -params['mean_viol'] / params['std_viol']
        else:
            raise NotImplementedError('This reward scaling does not exist!')

        # Potentially overwrite scaling with user settings
        if 'reward_factor' in params.keys():
            self.reward_factor = params['reward_factor']
        if 'reward_bias' in params.keys():
            self.reward_bias = params['reward_bias']
        if 'penalty_factor' in params.keys():
            self.penalty_factor = params['penalty_factor']
        if 'penalty_bias' in params.keys():
            self.penalty_bias = params['penalty_bias']

    def reset(self, step=None, test=False, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.info = {}
        self.step_in_episode = 0

        if not options:
            options = {}

        test = options.get('test', False)
        step = options.get('step', None)
        self.apply_action = options.get('new_action', True)

        if self.penalty_obs_space:
            # TODO: penalty obs currently only work with linear penalties
            if test and self.test_penalty is not None:
                self.linear_penalties = np.ones(
                    len(self.penalty_obs_space.low)) * self.test_penalty
            else:
                self.linear_penalties = self.penalty_obs_space.sample()
            self.volt_pen = {'linear_penalty': self.linear_penalties[0]}
            self.line_pen = {'linear_penalty': self.linear_penalties[1]}
            self.trafo_pen = {'linear_penalty': self.linear_penalties[2]}
            self.ext_grid_pen = {'linear_penalty': self.linear_penalties[3]}
            # TODO: How to deal with custom added penalties?!
            
        self._sampling(step, test, self.apply_action)
        
        if self.add_act_obs:
            # Use random actions as starting point so that agent learns to handle that
            # TODO: Maybe better to combine this with multi-step?!
            act = self.action_space.sample()
        else:
            # Reset all actions to default values
            act = (self.action_space.low + self.action_space.high) / 2
        self._apply_actions(act)

        if self.pf_for_obs is True:
            success = self._run_pf()
            if not success:
                logging.warning(
                    'Failed powerflow calculcation in reset. Try again!')
                return self.reset()

            self.prev_obj = self.calc_objective(self.net)
            self.prev_reward = self.calc_reward()

        return self._get_obs(self.obs_keys, self.add_time_obs), copy.deepcopy(self.info)

    def _sampling(self, step=None, test=False, sample_new=True, *args, **kwargs):
        data_distr = self.test_data if test is True else self.train_data
        kwargs.update(self.sampling_kwargs)

        # Maybe also allow different kinds of noise and similar! with `**sampling_params`?
        if data_distr == 'noisy_simbench' or 'noise_factor' in kwargs.keys():
            if sample_new:
                self._set_simbench_state(step, test, *args, **kwargs)
        elif data_distr == 'simbench':
            if sample_new:
                self._set_simbench_state(
                    step, test, noise_factor=0.0, *args, **kwargs)
        elif data_distr == 'full_uniform':
            self._sample_uniform(sample_new=sample_new)
        elif data_distr == 'normal_around_mean':
            self._sample_normal(sample_new=sample_new, *args, **kwargs)

    def _sample_uniform(self, sample_keys=None, sample_new=True):
        """ Standard pre-implemented method to set power system to a new random
        state from uniform sampling. Uses the observation space as basis.
        Requirement: For every observations there must be "min_{obs}" and
        "max_{obs}" given as range to sample from.
        """
        assert sample_new, 'Currently only implemented for sample_new=True'
        if not sample_keys:
            sample_keys = self.obs_keys
        for unit_type, column, idxs in sample_keys:
            if 'res_' not in unit_type:
                self._sample_from_range(unit_type, column, idxs)
        
    def _sample_from_range(self, unit_type, column, idxs):
        df = self.net[unit_type]
        # Make sure to sample from biggest possible range
        try:
            low = df[f'min_min_{column}'].loc[idxs]
        except KeyError:
            low = df[f'min_{column}'].loc[idxs]
        try:
            high = df[f'max_max_{column}'].loc[idxs]
        except KeyError:
            high = df[f'max_{column}'].loc[idxs]

        r = np.random.uniform(low, high, size=(len(idxs),))
        try:
            self.net[unit_type][column].loc[idxs] = r / df.scaling
        except AttributeError:
            # TODO: Add comment why this is necessary
            self.net[unit_type][column].loc[idxs] = r

    def _sample_normal(self, std=0.3, truncated=False, sample_new=True):
        """ Sample data around mean values from simbench data. """
        assert sample_new, 'Currently only implemented for sample_new=True'
        for unit_type, column, idxs in self.obs_keys:
            if 'res_' not in unit_type and 'poly_cost' not in unit_type:
                df = self.net[unit_type].loc[idxs]
                mean = df[f'mean_{column}']
                max_values = (df[f'max_max_{column}'] / df.scaling).to_numpy()
                min_values = (df[f'min_min_{column}'] / df.scaling).to_numpy()
                diff = max_values - min_values
                if truncated:
                    random_values = stats.truncnorm.rvs(
                        min_values, max_values, mean, std * diff, len(mean))
                else:
                    random_values = np.random.normal(
                        mean, std * diff, len(mean))
                    random_values = np.clip(
                        random_values, min_values, max_values)
                self.net[unit_type][column].loc[idxs] = random_values

    def _set_simbench_state(self, step: int=None, test=False,
                            noise_factor=0.1, noise_distribution='uniform',
                            *args, **kwargs):
        """ Standard pre-implemented method to sample a random state from the
        simbench time-series data and set that state.

        Works only for simbench systems!
        """

        total_n_steps = len(self.profiles[('load', 'q_mvar')])
        if step is None:
            if test is True and self.train_test_split is True:
                step = np.random.choice(self.test_steps)
            else:
                while True:
                    step = random.randint(0, total_n_steps - 1)
                    if self.train_test_split and step in self.test_steps:
                        continue
                    break
        else:
            assert step < total_n_steps

        self.current_step = step

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
                new_values = (data * noise).to_numpy()
            elif noise_distribution == 'normal':
                # Normal distribution: noise_factor as relative std deviation
                new_values = np.random.normal(
                    loc=data, scale=data.abs() * noise_factor)

            # Make sure that the range of original data remains unchanged
            # (Technical limits of the units remain the same)
            new_values = np.clip(
                new_values,
                self.profiles[type_act].min(
                )[self.net[unit_type].index].to_numpy(),
                self.profiles[type_act].max(
                )[self.net[unit_type].index].to_numpy())

            self.net[unit_type].loc[self.net[unit_type].index,
                                    actuator] = new_values

        return True

    def step(self, action, test=False, *args, **kwargs):
        assert not np.isnan(action).any()
        self.info = {}
        self.step_in_episode += 1

        if self.apply_action:
            self._apply_actions(action)
            success = self._run_pf()

            if not success:
                # Something went seriously wrong! Find out what!
                # Maybe NAN in power setpoints?!
                # Maybe simply catch this with a strong negative reward?!
                logging.critical(f'Powerflow not converged and reason unknown! Run diagnostic tool to at least find out what went wrong: {pp.diagnostic(self.net)}')
                raise pp.powerflow.LoadflowNotConverged()

        reward = self.calc_reward(test)

        if self.diff_reward:
            # Do not use the objective as reward, but their diff instead
            reward -= self.prev_reward

        if self.steps_per_episode == 1:
            terminated = True
            truncated = False
        elif self.step_in_episode >= self.steps_per_episode:
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        obs = self._get_obs(self.obs_keys, self.add_time_obs)
        assert not np.isnan(obs).any()

        return obs, reward, terminated, truncated, copy.deepcopy(self.info)

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

        self._autocorrect_apparent_power(self.priority)

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

    def _run_pf(self, enforce_q_lims=True, calculate_voltage_angles=False, voltage_depend_loads=False, **kwargs):
        try:
            pp.runpp(self.net,
                     voltage_depend_loads=voltage_depend_loads,
                     enforce_q_lims=enforce_q_lims,
                     calculate_voltage_angles=calculate_voltage_angles,
                     **kwargs)

        except pp.powerflow.LoadflowNotConverged:
            logging.warning('Powerflow not converged!!!')
            return False
        return True

    def calc_objective(self, net):
        """ Default: Compute reward/costs from poly costs. Works only if
        defined as pandapower OPF problem and only for poly costs! If that is
        not the case, this method needs to be overwritten! """
        return -min_pp_costs(net)

    def calc_violations(self):
        """ Constraint violations result in a penalty that can be subtracted
        from the reward.
        Standard penalties: voltage band, overload of lines & transformers. """

        valids_violations_penalties = [
            voltage_violation(self.net, self.autoscale_violations,
                **self.volt_pen),
            line_overload(self.net, self.autoscale_violations,
                **self.line_pen),
            trafo_overload(self.net, self.autoscale_violations,
                **self.trafo_pen),
            ext_grid_overpower(self.net, 'q_mvar', self.autoscale_violations,
                **self.ext_grid_pen),
            ext_grid_overpower(self.net, 'p_mw', self.autoscale_violations,
                **self.ext_grid_pen)]

        valids, viol, penalties = zip(*valids_violations_penalties)

        return np.array(valids), np.array(viol), np.array(penalties)

    def calc_reward(self, test=False):
        """ Combine objective function and the penalties together. """
        objectives = self.calc_objective(self.net)
        self.info['objectives'] = objectives
        valids, viol, penalties = self.calc_violations()

        self.info['valids'] = np.array(valids)
        self.info['violations'] = np.array(viol)
        self.info['unscaled_penalties'] = np.array(penalties)

        # TODO: re-structure this whole reward calculation?!
        if self.reward_function == 'summation':
            # Idea: Add penalty to objective function (no change required)
            pass
        elif self.reward_function == 'replacement':
            # Idea: Only give objective as reward, if solution valid
            if not valids.all():
                objectives[:] = 0.0
            else:
                # Make sure that the objective is always positive
                # This way, even the worst-case results in zero reward
                objectives += self.mean_abs_obj / len(objectives)
        elif self.reward_function == 'replacementA':
            # Idea: Only give objective as reward, if solution valid
            if not valids.all():
                objectives[:] = 0.0
            else:
                # Use the worst case objective instead here 
                objectives += abs(self.min_obj) / len(objectives)
        # elif self.reward_function == 'replacement_plus_summation':
        #     # TODO Idea: can these two be combined?!
        #     # If valid: Use objective as reward
        #     # If invalid: Use penalties as reward + objective to make sure agent learns both at the same time (and not first only penalties and then only objective))
        #     # But ensure that the reward is always higher if valid compared to invalid 
        #     if valids.all():
        #         objectives += abs(self.min_obj)
        elif self.reward_function == "flat":
            if not valids.all():
                # Make sure the penalty is always bigger than the objective
                penalties = 2 * -abs(sum(objectives))
                self.info['penalties'] = penalties
        else:
            raise NotImplementedError('This reward definition does not exist!')

        # full_obj = np.append(objectives, penalties)
        
        obj = sum(objectives)
        pen = sum(penalties)
        # Perform reward scaling
        obj = obj * self.reward_factor + self.reward_bias
        pen = pen * self.penalty_factor + self.penalty_bias

        self.info['cost'] = abs(pen)  # Standard cost definition in Safe RL

        # TODO: Evaluate if this makes any sense at all
        # if self.autoscale_penalty:
        #     # Scale the penalty with the objective function
        #     if self.min_penalty:
        #         # The min_penalty prevents the penalty to become ~0.0
        #         # TODO: Maybe a simple sqrt is better?!
        #         penalties *= max(abs(sum(objectives)), self.min_penalty)
        #     else:
        #         penalties *= abs(sum(objectives))
        #     self.info['penalties'] = penalties

        # if not self.vector_reward:
        #     # Use scalar reward 
        #     reward = sum(full_obj)
        # else:
        #     # Reward as a numpy array
        #     reward = full_obj

        if self.penalty_weight is not None:
            reward = obj * (1 - self.penalty_weight) + pen * self.penalty_weight
        else:
            reward = obj + pen

        if self.squash_reward and not test:
            reward = np.sign(reward) * np.log(np.abs(reward) + 1)
            if self.min_penalty and not valids.all():
                # Prevent that the penalty gets completely squashed
                reward -= self.min_penalty

        return reward

    def _get_obs(self, obs_keys, add_time_obs):
        obss = [(self.net[unit_type][column].loc[idxs].to_numpy())
                if (unit_type != 'load' or not self.bus_wise_obs)
                else get_bus_aggregated_obs(self.net, 'load', column, idxs)
                for unit_type, column, idxs in obs_keys]

        if self.penalty_obs_space:
            obss = [self.linear_penalties] + obss

        if add_time_obs:
            time_obs = get_simbench_time_observation(
                self.profiles, self.current_step)
            obss = [time_obs] + obss

        return np.concatenate(obss)

    def render(self):
        logging.warning(f'Rendering not supported!')

    def get_current_actions(self):
        # Attention: These are not necessarily the actions of the RL agent
        # because some re-scaling might have happened!
        action = [(self.net[f'res_{unit_type}'][column].loc[idxs]
                   / self.net[unit_type][f'max_max_{column}'].loc[idxs])
                  for unit_type, column, idxs in self.act_keys]
        return np.concatenate(action)

    def baseline_reward(self, **kwargs):
        """ Compute some baseline to compare training performance with. In this
        case, use the optimal possible reward, which can be computed with the
        optimal power flow. """
        success = self._optimal_power_flow(**kwargs)
        if not success:
            return np.nan
        objectives = self.calc_objective(self.net)
        valids, violations, penalties = self.calc_violations()
        logging.info(f'Optimal violations: {violations}')
        logging.info(f'Baseline actions: {self.get_current_actions()}')
        if sum(penalties) > 0:
            logging.warning(f'There are baseline penalties: {penalties}'
                            f' with violations: {violations}'
                            '(should normally not happen! Check if this is some'
                            'special case with soft constraints!')

        return sum(np.append(objectives, penalties))

    def _optimal_power_flow(self, **kwargs):
        try:
            pp.runopp(self.net, calculate_voltage_angles=False, **kwargs)
        except pp.optimal_powerflow.OPFNotConverged:
            logging.warning('OPF not converged!!!')
            return False
        return True


def get_obs_space(net, obs_keys: list, add_time_obs: bool, 
                  penalty_obs_space: gym.Space=None, seed: int=None,
                  last_n_obs: int=1, bus_wise_obs=False):
    """ Get observation space from the constraints of the power network. """
    lows, highs = [], []

    if add_time_obs:
        # Time is always given as observation of lenght 6 in range [-1, 1]
        # at the beginning of the observation!
        lows.append(-np.ones(6))
        highs.append(np.ones(6))

    if penalty_obs_space:
        # Add penalty observation space
        lows.append(penalty_obs_space.low)
        highs.append(penalty_obs_space.high)

    for unit_type, column, idxs in obs_keys:
        if 'res_' in unit_type:
            # The constraints are never defined in the results table
            unit_type = unit_type[4:]

        try:
            if f'min_min_{column}' in net[unit_type].columns:
                l = net[unit_type][f'min_min_{column}'].loc[idxs].to_numpy()
            else:
                l = net[unit_type][f'min_{column}'].loc[idxs].to_numpy()
            if f'max_max_{column}' in net[unit_type].columns:
                h = net[unit_type][f'max_max_{column}'].loc[idxs].to_numpy()
            else:
                h = net[unit_type][f'max_{column}'].loc[idxs].to_numpy()
        except KeyError:
            # Special case: trafos and lines (have minimum constraint of zero)
            l = np.zeros(len(idxs))
            # Assumption: No lines with loading more than 150%
            h = net[unit_type][f'max_{column}'].loc[idxs].to_numpy() * 1.5

        # Special case: voltages
        if column == 'vm_pu' or unit_type == 'ext_grid':
            diff = h - l
            # Assumption: If [0.95, 1.05] voltage band, no voltage outside [0.875, 1.125] range
            l = l - diff * 0.75
            h = h + diff * 0.75
            
        try:
            if 'min' in column or 'max' in column:
                # Constraints need to remain scaled
                raise AttributeError
            l = l / net[unit_type].scaling.loc[idxs].to_numpy()
            h = h / net[unit_type].scaling.loc[idxs].to_numpy()
        except AttributeError:
            logging.info(
                f'Scaling for {unit_type} not defined: assume scaling=1')

        if bus_wise_obs and unit_type == 'load':
            # Aggregate loads bus-wise. Currently only for loads!
            buses = sorted(set(net[unit_type].bus))
            l = [sum(l[net[unit_type].bus == bus]) for bus in buses]
            h = [sum(h[net[unit_type].bus == bus]) for bus in buses]

        for _ in range(last_n_obs):
            lows.append(l)
            highs.append(h)

    assert not sum(pd.isna(l).any() for l in lows)
    assert not sum(pd.isna(h).any() for h in highs)

    return gym.spaces.Box(
        np.concatenate(lows, axis=0), np.concatenate(highs, axis=0), seed=seed)


def get_action_space(act_keys: list, seed: int):
    """ Get RL action space from defined actuators. """
    low = np.array([])
    high = np.array([])
    for unit_type, column, idxs in act_keys:
        # Storages in general and reactive power can take negative values
        condition = (unit_type == 'storage' or column == 'q_mvar')
        new_lows = -np.ones(len(idxs)) if condition else np.zeros(len(idxs))

        low = np.append(low, new_lows)
        high = np.append(high, np.ones(len(idxs)))

    return gym.spaces.Box(low, high, seed=seed)


def define_test_steps(test_share=0.2):
    """ Return the indices of the simbench test data points """
    assert test_share > 0.0, 'Please set train_test_split=False if no separate test data should be used'

    if test_share == 1.0:
        # Special case: Use the full simbench data set as test set
        return np.arange(24 * 4 * 366)

    # Use weekly blocks to make sure that all weekdays are equally represented
    n_weeks = int(52 * test_share)
    # Sample equidistant weeks from the whole year
    week_idxs = np.linspace(0, 52, num=n_weeks, endpoint=False, dtype=int)

    one_week = 7 * 24 * 4
    return np.concatenate(
        [np.arange(idx * one_week, (idx + 1) * one_week) for idx in week_idxs]
    )


def get_simbench_time_observation(profiles: dict, current_step: int):
    """ Return current time in sinus/cosinus form.
    Example daytime: (0.0, 1.0) = 00:00 and (1.0, 0.0) = 06:00. Idea from
    https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
    """
    total_n_steps = len(profiles[('load', 'q_mvar')])
    # number of steps per timeframe
    dayly, weekly, yearly = (24 * 4, 7 * 24 * 4, total_n_steps)
    time_obs = []
    for timeframe in (dayly, weekly, yearly):
        timestep = current_step % timeframe
        cyclical_time = 2 * np.pi * timestep / timeframe
        time_obs.append(np.sin(cyclical_time))
        time_obs.append(np.cos(cyclical_time))

    return np.array(time_obs)


def get_bus_aggregated_obs(net, unit_type, column, idxs):
    """ Aggregate power values that are connected to the same bus to reduce
    state space. """
    df = net[unit_type].iloc[idxs]
    return df.groupby(['bus'])[column].sum().to_numpy()
