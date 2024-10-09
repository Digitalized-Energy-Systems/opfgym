
import abc
import copy
import logging
import warnings

import gymnasium as gym
import numpy as np
import pandapower as pp
import pandas as pd
import scipy
from scipy import stats
from typing import Tuple

from opfgym.penalties import (voltage_violation, line_overload,
                             trafo_overload, ext_grid_overpower)
from opfgym.objectives import min_pp_costs
from opfgym.util.normalization import get_normalization_params
from opfgym.simbench.data_split import define_test_train_split
from opfgym.simbench.time_observation import get_simbench_time_observation

warnings.simplefilter('once')


class PowerFlowNotAvailable(Exception):
    pass

class OpfEnv(gym.Env, abc.ABC):
    def __init__(self,
                 evaluate_on='validation',
                 steps_per_episode=1,
                 autocorrect_prio='p_mw',
                 pf_for_obs=None,
                 bus_wise_obs=False,
                 diff_objective=False,
                 reward_function: str='summation',
                 reward_function_params: dict=None,
                 clip_reward: tuple=None,
                 reward_scaling: str=None,
                 reward_scaling_params: dict=None,
                 remove_normal_obs=False,
                 add_res_obs=False,
                 add_time_obs=False,
                 add_act_obs=False,
                 add_mean_obs=False,
                 train_data='simbench',
                 test_data='simbench',
                 sampling_kwargs: dict=None,
                 volt_pen_kwargs: dict=None,
                 line_pen_kwargs: dict=None,
                 trafo_pen_kwargs: dict=None,
                 ext_grid_pen_kwargs: dict=None,
                 only_worst_case_violations=False,
                 autoscale_violations=True,
                 penalty_weight=0.5,
                 penalty_obs_range: tuple=None,
                 test_penalty=None,
                 autoscale_actions=True,
                 diff_action_step_size=None,
                 clipped_action_penalty=0,
                 initial_action='center',
                 seed=None,
                 *args, **kwargs):

        self.evaluate_on = evaluate_on
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
        if add_res_obs is True:
            # Default: Add all results that are usually available
            add_res_obs = ('voltage_magnitude', 'voltage_angle', 
                           'line_loading', 'trafo_loading', 'ext_grid_power')
        if add_res_obs:
            # Tricky: Only use buses with actual units connected. Otherwise, too many auxiliary buses are included.
            bus_idxs = set(self.net.load.bus) | set(self.net.sgen.bus) | set(self.net.gen.bus) | set(self.net.storage.bus)
            add_obs = []
            if 'voltage_magnitude' in add_res_obs:
                add_obs.append(('res_bus', 'vm_pu', np.sort(list(bus_idxs))))
            if 'voltage_angle' in add_res_obs:
                add_obs.append(('res_bus', 'va_degree', np.sort(list(bus_idxs))))
            if 'line_loading' in add_res_obs:
                add_obs.append(('res_line', 'loading_percent', self.net.line.index))
            if 'trafo_loading' in add_res_obs:
                add_obs.append(('res_trafo', 'loading_percent', self.net.trafo.index))
            if 'ext_grid_power' in add_res_obs:
                add_obs.append(('res_ext_grid', 'p_mw', self.net.ext_grid.index))
                add_obs.append(('res_ext_grid', 'q_mvar', self.net.ext_grid.index))
            self.obs_keys.extend(add_obs)   

        self.add_mean_obs = add_mean_obs

        if penalty_obs_range:
            n_penalties = 4 # TODO
            self.penalty_obs_space = gym.spaces.Box(
                low=np.ones(n_penalties) * penalty_obs_range[0], 
                high=np.ones(n_penalties) * penalty_obs_range[1], 
                seed=seed)
            self.test_penalty = test_penalty
        else:
            self.penalty_obs_space = None

        # Define observation and action space
        self.bus_wise_obs = bus_wise_obs
        self.observation_space = get_obs_space(
            self.net, self.obs_keys, add_time_obs, add_mean_obs, 
            self.penalty_obs_space, seed, bus_wise_obs=bus_wise_obs)
        n_actions = sum([len(idxs) for _, _, idxs in self.act_keys])
        self.action_space = gym.spaces.Box(0, 1, shape=(n_actions,), seed=seed)

        # Reward function
        self.reward_function = reward_function
        self.reward_function_params = reward_function_params if reward_function_params else {}
        self.volt_pen = volt_pen_kwargs if volt_pen_kwargs else {}
        self.line_pen = line_pen_kwargs if line_pen_kwargs else {}
        self.trafo_pen = trafo_pen_kwargs if trafo_pen_kwargs else {}
        self.ext_grid_pen = ext_grid_pen_kwargs if ext_grid_pen_kwargs else {}
        self.only_worst_case_violations = only_worst_case_violations
        self.autoscale_violations = autoscale_violations
        self.clip_reward = clip_reward

        # Action space details
        self.priority = autocorrect_prio
        self.autoscale_actions = autoscale_actions
        self.diff_action_step_size = diff_action_step_size
        self.clipped_action_penalty = clipped_action_penalty
        self.initial_action = initial_action

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

        self.diff_objective = diff_objective
        if diff_objective:
            # An initial power flow is required to compute the initial objective
            self.pf_for_obs = True

        self.test_steps, self.validation_steps, self.train_steps = define_test_train_split(**kwargs)

        # Prepare reward scaling for later on 
        self.reward_scaling = reward_scaling
        self.penalty_weight = penalty_weight
        reward_scaling_params = reward_scaling_params if reward_scaling_params else {}
        if reward_scaling_params == 'auto' or (
                'num_samples' in reward_scaling_params) or (
                not reward_scaling_params and reward_scaling):
            # Find reward range by trial and error
            params = get_normalization_params(self, **reward_scaling_params)
        else:
            params = reward_scaling_params

        self.normalization_params = params
        if not reward_scaling:
            self.objective_factor = 1
            self.objective_bias = 0
            self.penalty_factor = 1
            self.penalty_bias = 0
        elif reward_scaling == 'minmax':
            # Scale from range [min, max] to range [-1, 1]
            # formula: (obj - min_obj) / (max_obj - min_obj) * 2 - 1
            diff = (params['max_obj'] - params['min_obj']) / 2
            self.objective_factor = 1 / diff
            self.objective_bias = -(params['min_obj'] / diff + 1)
            diff = 2 * (params['max_viol'] - params['min_viol'])
            self.penalty_factor = 1 / diff
            self.penalty_bias = -(params['min_viol'] / diff + 1)
        elif reward_scaling == 'normalization':
            # Scale so that mean is zero and standard deviation is one
            # formula: (obj - mean_obj) / obj_std
            self.objective_factor = 1 / params['std_obj']
            self.objective_bias = -params['mean_obj'] / params['std_obj']
            self.penalty_factor = 1 / params['std_viol']
            self.penalty_bias = -params['mean_viol'] / params['std_viol']
        else:
            raise NotImplementedError('This reward scaling does not exist!')

        # Error handling
        if np.isnan(self.penalty_bias):
            self.penalty_bias = 0
        if np.isinf(self.penalty_factor):
            self.penalty_factor = 1

        # Potentially overwrite scaling with user settings
        if 'reward_factor' in params.keys():
            self.objective_factor = params['reward_factor']
        if 'objective_bias' in params.keys():
            self.objective_bias = params['objective_bias']
        if 'penalty_factor' in params.keys():
            self.penalty_factor = params['penalty_factor']
        if 'penalty_bias' in params.keys():
            self.penalty_bias = params['penalty_bias']

        if self.reward_function in ('replacement', 'parameterized'):
            valid_reward = self.reward_function_params.get('valid_reward', 1)
            # Standard variants: Use mean or worst case objective as reward
            if isinstance(valid_reward, str):
                if valid_reward == 'worst':
                    valid_reward = self.normalization_params['min_obj']
                elif valid_reward == 'mean':
                    valid_reward = self.normalization_params['mean_obj']
                valid_reward = valid_reward * self.objective_factor + self.objective_bias
            self.valid_reward = valid_reward

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)
        self.info = {}
        self.current_simbench_step = None
        self.step_in_episode = 0

        if not options:
            options = {}

        self.test = options.get('test', False)
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

        self._sampling(step, self.test, self.apply_action)
        self.power_flow_available = False

        if self.initial_action == 'random':
            # Use random actions as starting point so that agent learns to handle that
            act = self.action_space.sample()
        else:
            # Reset all actions to default values
            act = (self.action_space.low + self.action_space.high) / 2
        self._apply_actions(act)

        if self.pf_for_obs is True:
            success = self._run_power_flow()
            if not success:
                logging.warning(
                    'Failed powerflow calculcation in reset. Try again!')
                return self.reset()

            self.initial_obj = self.calc_objective(base_objective=True)

        return self._get_obs(self.obs_keys, self.add_time_obs), copy.deepcopy(self.info)

    def _sampling(self, step=None, test=False, sample_new=True, 
                  *args, **kwargs) -> None:
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
            self._sample_normal(sample_new=sample_new, **kwargs)
        elif data_distr == 'mixed':
            # Use different data sources with different probabilities
            r = np.random.random()
            data_probs = kwargs.get('data_probabilities', (0.5, 0.75, 1.0))
            if r < data_probs[0]:
                self._set_simbench_state(step, test, *args, **kwargs)
            elif r < data_probs[1]:
                self._sample_uniform(sample_new=sample_new)
            else:
                self._sample_normal(sample_new=sample_new, **kwargs)

    def _sample_uniform(self, sample_keys=None, sample_new=True) -> None:
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

    def _sample_from_range(self, unit_type, column, idxs) -> None:
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
            # Constraints are scaled, which is why we need to divide by scaling
            self.net[unit_type][column].loc[idxs] = r / df.scaling[idxs]
        except AttributeError:
            # If scaling factor is not defined, assume scaling=1
            self.net[unit_type][column].loc[idxs] = r

    def _sample_normal(self, relative_std=None, truncated=False,
                       sample_new=True, **kwargs) -> None:
        """ Sample data around mean values from simbench data. """
        assert sample_new, 'Currently only implemented for sample_new=True'
        for unit_type, column, idxs in self.obs_keys:
            if 'res_' in unit_type or 'poly_cost' in unit_type:
                continue 

            df = self.net[unit_type].loc[idxs]
            mean = df[f'mean_{column}']

            max_values = (df[f'max_max_{column}'] / df.scaling).to_numpy()
            min_values = (df[f'min_min_{column}'] / df.scaling).to_numpy()
            diff = max_values - min_values
            if relative_std:
                std = relative_std * diff
            else:
                std = df[f'std_dev_{column}']

            if truncated:
                # Make sure to re-distribute truncated values 
                random_values = stats.truncnorm.rvs(
                    min_values, max_values, mean, std * diff, len(mean))
            else:
                # Simply clip values to min/max range
                random_values = np.random.normal(
                    mean, std * diff, len(mean))
                random_values = np.clip(
                    random_values, min_values, max_values)
            self.net[unit_type][column].loc[idxs] = random_values

    def _set_simbench_state(self, step: int=None, test=False,
                            noise_factor=0.1, noise_distribution='uniform',
                            in_between_steps=False, *args, **kwargs) -> None:
        """ Standard pre-implemented method to sample a random state from the
        simbench time-series data and set that state.

        Works only for simbench systems!
        """

        total_n_steps = len(self.profiles[('load', 'q_mvar')])
        if step is None:
            if test is True and self.evaluate_on == 'test':
                step = np.random.choice(self.test_steps)
            elif test is True and self.evaluate_on == 'validation':
                step = np.random.choice(self.validation_steps)
            else:
                step = np.random.choice(self.train_steps)
        else:
            assert step < total_n_steps

        self.current_simbench_step = step

        for type_act in self.profiles.keys():
            if not self.profiles[type_act].shape[1]:
                continue
            unit_type, actuator = type_act
            data = self.profiles[type_act].loc[step, self.net[unit_type].index]

            if in_between_steps and step < total_n_steps - 1:
                # Random linear interpolation between two steps
                next_data = self.profiles[type_act].loc[step + 1, self.net[unit_type].index]
                r = np.random.random()
                data = data * r + next_data * (1 - r)

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

    def step(self, action, *args, **kwargs) -> tuple:
        assert not np.isnan(action).any()
        self.info = {}
        self.step_in_episode += 1

        if self.apply_action:
            correction = self._apply_actions(action, self.diff_action_step_size)
            success = self._run_power_flow()

            if not success:
                # Something went seriously wrong! Find out what!
                # Maybe NAN in power setpoints?!
                # Maybe simply catch this with a strong negative reward?!
                logging.critical(f'\nPowerflow not converged and reason unknown! Run diagnostic tool to at least find out what went wrong: {pp.diagnostic(self.net)}')
                
                self.info['valids'] = np.array([False] * 5)
                self.info['violations'] = np.array([1] * 5)
                self.info['unscaled_penalties'] = np.array([1] * 5)
                self.info['penalty'] = 5
                return np.array([np.nan]), np.nan, True, False, copy.deepcopy(self.info)

        reward = self.calc_reward()

        if self.clipped_action_penalty and self.apply_action:
            reward -= correction * self.clipped_action_penalty

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

    def _apply_actions(self, action, diff_action_step_size=None) -> float:
        """ Apply agent actions as setpoints to the power system at hand. 
        Returns the mean correction that was necessary to make the actions
        valid."""

        # Clip invalid actions
        action = np.clip(action, self.action_space.low, self.action_space.high)

        counter = 0
        for unit_type, actuator, idxs in self.act_keys:
            if len(idxs) == 0:
                continue

            df = self.net[unit_type]
            partial_act = action[counter:counter + len(idxs)]


            if self.autoscale_actions:
                # Ensure that actions are always valid by using the current range
                min_action = df[f'min_{actuator}'].loc[idxs]
                max_action = df[f'max_{actuator}'].loc[idxs]
            else:
                # Use the full action range instead (only different if min/max change during training)
                min_action = df[f'min_min_{actuator}'].loc[idxs]
                max_action = df[f'max_max_{actuator}'].loc[idxs]

            delta_action = (max_action - min_action).values

            # Always use continuous action space [0, 1]
            if diff_action_step_size:
                # Agent sets incremental setpoints instead of absolute ones.
                previous_setpoints = self.net[unit_type][actuator].loc[idxs].values
                if 'scaling' in df.columns:
                    previous_setpoints *= df.scaling.loc[idxs]
                # Make sure decreasing the setpoint is possible as well
                partial_act = partial_act * 2 - 1
                setpoints = partial_act * diff_action_step_size * delta_action + previous_setpoints
            else:
                # Agent sets absolute setpoints in range [min, max]
                setpoints = partial_act * delta_action + min_action

            # Autocorrect impossible setpoints
            if not self.autoscale_actions or diff_action_step_size:
                if f'max_{actuator}' in df.columns:
                    mask = setpoints > df[f'max_{actuator}'].loc[idxs]
                    setpoints[mask] = df[f'max_{actuator}'].loc[idxs][mask]
                if f'min_{actuator}' in df.columns:
                    mask = setpoints < df[f'min_{actuator}'].loc[idxs]
                    setpoints[mask] = df[f'min_{actuator}'].loc[idxs][mask]

            if 'scaling' in df.columns:
                # Scaling column sometimes not existing
                setpoints /= df.scaling.loc[idxs]

            if actuator == 'closed' or actuator == 'in_service':
                # Special case: Only binary actions
                setpoints = np.round(setpoints).astype(bool)
            elif actuator == 'tap_pos' or actuator == 'step':
                # Special case: Only discrete actions
                setpoints = np.round(setpoints)

            self.net[unit_type][actuator].loc[idxs] = setpoints

            counter += len(idxs)

        # TODO: Not really relevant if active/reactive not optimized together
        # self._autocorrect_apparent_power(self.priority)

        # Did the action need to be corrected to be in bounds?
        mean_correction = np.mean(abs(self.get_current_actions(False) - action))

        return mean_correction

    def _autocorrect_apparent_power(self, priority='p_mw') -> float:
        """ Autocorrect to maximum apparent power if necessary. Relevant for
        sgens, loads, and storages """
        not_prio = 'p_mw' if priority == 'q_mvar' else 'q_mvar'
        correction = 0
        for unit_type in ('sgen', 'load', 'storage'):
            df = self.net[unit_type]
            if 'max_s_mva' in df.columns:
                s_mva2 = df.max_s_mva.to_numpy() ** 2
                values2 = (df[priority] * df.scaling).to_numpy() ** 2
                # Make sure to prevent negative values for sqare root
                max_values = np.maximum(s_mva2 - values2, 0)**0.5 / df.scaling
                # Reduce non-priority power setpoints
                new_values = np.sign(df[not_prio]) * np.minimum(df[not_prio].abs(), max_values)
                correction += (self.net[unit_type][not_prio] - new_values).abs().sum()
                self.net[unit_type][not_prio] = new_values

        return correction

    def calc_objective(self, base_objective=True) -> np.ndarray:
        """ Default: Compute reward/costs from poly costs. Works only if
        defined as pandapower OPF problem and only for poly costs! If that is
        not the case, this method needs to be overwritten! """
        if base_objective or not self.diff_objective:
            return -min_pp_costs(self.net)
        else:
            return -min_pp_costs(self.net) - self.initial_obj

    def calc_violations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Constraint violations result in a penalty that can be subtracted
        from the reward.
        Standard penalties: voltage band, overload of lines & transformers. """

        valids_violations_penalties = [
            voltage_violation(self.net, self.autoscale_violations,
                worst_case_only=self.only_worst_case_violations, **self.volt_pen),
            line_overload(self.net, self.autoscale_violations,
                worst_case_only=self.only_worst_case_violations, **self.line_pen),
            trafo_overload(self.net, self.autoscale_violations,
                worst_case_only=self.only_worst_case_violations, **self.trafo_pen),
            ext_grid_overpower(self.net, 'q_mvar', self.autoscale_violations,
                worst_case_only=self.only_worst_case_violations, **self.ext_grid_pen),
            ext_grid_overpower(self.net, 'p_mw', self.autoscale_violations,
                worst_case_only=self.only_worst_case_violations, **self.ext_grid_pen)]

        valids, viol, penalties = zip(*valids_violations_penalties)

        return np.array(valids), np.array(viol), np.array(penalties)

    def calc_reward(self) -> float:
        """ Combine objective function and the penalties together. """
        objective = sum(self.calc_objective(base_objective=False))
        valids, violations, penalties = self.calc_violations()

        penalty = sum(penalties)

        # Perform potential reward clipping (e.g., to prevent exploding rewards)
        try:
            if self.normalization_params['clip_range_obj'][0] is not None:
                objective = np.clip(objective,
                                    self.normalization_params['clip_range_obj'][0],
                                    self.normalization_params['clip_range_obj'][1])
        except KeyError:
            pass
        try:
            if self.normalization_params['clip_range_viol'][0] is not None:
                penalty = np.clip(penalty,
                                self.normalization_params['clip_range_viol'][0],
                                self.normalization_params['clip_range_viol'][1])
        except KeyError:
            pass

        # Perform reward scaling, e.g., to range [-1, 1] (if defined)
        objective = objective * self.objective_factor + self.objective_bias
        penalty = penalty * self.penalty_factor + self.penalty_bias

        self.info['valids'] = np.array(valids)
        self.info['violations'] = np.array(violations)  
        self.info['unscaled_penalties'] = np.array(penalties)
        self.info['penalty'] = penalty
        # Standard cost definition in Safe RL (Do not use bias here to prevent sign flip)
        if not valids.all():
            self.info['cost'] = abs(penalty * self.penalty_factor - self.reward_function_params.get('invalid_penalty', 0.0))  
        else:
            self.info['cost'] = 0

        if self.reward_function == 'summation':
            # Add penalty to objective function (no change required)
            pass
        elif self.reward_function == 'replacement':
            # Only give objective as reward, if solution valid
            if valids.all():
                # Make sure that the valid reward is always higher
                objective += self.valid_reward
            else:
                objective = 0.0
        elif self.reward_function == "parameterized":
            # Parameterized combination of summation and replacement.
            # If valid_reward==0 & objective_share==1: Summation reward
            # If valid_reward>0 & objective_share==0: Replacement reward
            # The range in between represents weighted combinations of both
            # The invalid_penalty is added to allow for inverse replacement method
            # (punish invalid states instead of rewarding valid ones)
            if valids.all():
                # Positive penalty may sound strange but the intution is that 
                # the penalty reward term represents constraint satisfaction
                penalty += self.reward_function_params.get('valid_reward', 0)
            else:
                objective *= self.reward_function_params.get('objective_share', 1)
                penalty -= self.reward_function_params.get('invalid_penalty', 0.5)
        else:
            raise NotImplementedError('This reward definition does not exist!')

        if self.penalty_weight is not None:
            reward = objective * (1 - self.penalty_weight) + penalty * self.penalty_weight
        else:
            reward = objective + penalty

        if self.clip_reward:
            reward = np.clip(reward, self.clip_reward[0], self.clip_reward[1])

        return reward

    def _get_obs(self, obs_keys, add_time_obs) -> np.ndarray:
        obss = [(self.net[unit_type][column].loc[idxs].to_numpy())
                if (unit_type != 'load' or not self.bus_wise_obs)
                else get_bus_aggregated_obs(self.net, 'load', column, idxs)
                for unit_type, column, idxs in obs_keys]

        if self.penalty_obs_space:
            obss = [self.linear_penalties] + obss

        if self.add_mean_obs:
            mean_obs = [np.mean(partial_obs) for partial_obs in obss 
                        if len(partial_obs) > 1]
            obss.append(mean_obs)

        if add_time_obs and self.current_simbench_step is not None:
            time_obs = get_simbench_time_observation(
                self.profiles, self.current_simbench_step)
            obss = [time_obs] + obss

        return np.concatenate(obss)

    def render(self, **kwargs):
        """ Render the current state of the power system. Uses the `simple_plot` 
        pandapower method. Overwrite for more sophisticated rendering. For 
        kwargs, refer to the pandapower docs: 
        https://pandapower.readthedocs.io/en/latest/plotting/matplotlib/simple_plot.html"""
        ax = pp.plotting.simple_plot(self.net, **kwargs)
        return ax

    def get_current_actions(self, results=True) -> np.ndarray:
        # Attention: These are not necessarily the actions of the RL agent
        # because some re-scaling might have happened!
        # These are the actions from the original action space [0, 1]
        res_flag = 'res_' if results else ''
        action = []
        for unit_type, column, idxs in self.act_keys:
            setpoints = self.net[f'{res_flag}{unit_type}'][column].loc[idxs]

            # If data not taken from res table, scaling required
            if not results and 'scaling' in self.net[unit_type].columns:
                setpoints *= self.net[unit_type].scaling.loc[idxs]

            # Action space depends on autoscaling 
            min_id = 'min_' if self.autoscale_actions else 'min_min_'
            max_id = 'max_' if self.autoscale_actions else 'max_max_' 
            min_values = self.net[unit_type][f'{min_id}{column}'].loc[idxs]
            max_values = self.net[unit_type][f'{max_id}{column}'].loc[idxs]

            action.append((setpoints - min_values) / (max_values - min_values))

        action = np.concatenate(action)

        return action

    def is_valid(self) -> bool:
        """ Return True if the current state satisfies all constraints. """
        if not self.power_flow_available:
            self._run_power_flow()
        valids, _, _ = self.calc_violations()
        return valids.all()

    def baseline_objective(self, **kwargs) -> float:
        """ Compute the optimal system state via pandapower optimal power flow
        and return the objective function value of that state.
        Warning: Changes the state of the underlying power system! """
        success = self._run_optimal_power_flow(**kwargs)
        if not success:
            return np.nan
        objectives = self.calc_objective(base_objective=True)
        valids, violations, penalties = self.calc_violations()
        logging.info(f'Optimal violations: {violations}')
        logging.info(f'Baseline actions: {self.get_current_actions()}')
        if sum(penalties) > 0:
            logging.warning(f'There are baseline penalties: {penalties}'
                            f' with violations: {violations}'
                            '(should normally not happen! Check if this is some'
                            'special case with soft constraints!')

        return sum(np.append(objectives, penalties))

    def _run_optimal_power_flow(self, calculate_voltage_angles=False, **kwargs):
        self.power_flow_available = True
        try:
            pp.runopp(self.net,
                      calculate_voltage_angles=calculate_voltage_angles,
                      **kwargs)
        except pp.optimal_powerflow.OPFNotConverged:
            logging.warning('OPF not converged!!!')
            return False
        return True

    def _run_power_flow(self, enforce_q_lims=True,
                        calculate_voltage_angles=False,
                        voltage_depend_loads=False,
                        **kwargs):
        self.power_flow_available = True
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


def get_obs_space(net, obs_keys: list, add_time_obs: bool, 
                  add_mean_obs: bool=False, penalty_obs_space: gym.Space=None, 
                  seed: int=None, last_n_obs: int=1, bus_wise_obs=False
                  ) -> gym.spaces.Box:
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

        if column == 'va_degree':
            # Usually no constraints for voltage angles defined
            # Assumption: [30, 30] degree range (experience)
            l = np.full(len(idxs), -30)
            h = np.full(len(idxs), +30)
        else:
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

    if add_mean_obs:
        # Add mean values of each category as additional observations
        start_from = 1 if add_time_obs else 0
        add_l = [np.mean(l) for l in lows[start_from:] if len(l) > 1]
        add_h = [np.mean(h) for h in highs[start_from:] if len(h) > 1]
        lows.append(np.array(add_l))
        highs.append(np.array(add_h))

    assert not sum(pd.isna(l).any() for l in lows)
    assert not sum(pd.isna(h).any() for h in highs)

    return gym.spaces.Box(
        np.concatenate(lows, axis=0), np.concatenate(highs, axis=0), seed=seed)


def get_bus_aggregated_obs(net, unit_type, column, idxs) -> np.ndarray:
    """ Aggregate power values that are connected to the same bus to reduce
    state space. """
    df = net[unit_type].iloc[idxs]
    return df.groupby(['bus'])[column].sum().to_numpy()
