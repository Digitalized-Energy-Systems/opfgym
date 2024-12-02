
from collections.abc import Callable
import copy
import logging
import inspect

import gymnasium as gym
import numpy as np
import pandapower as pp
import pandas as pd
import scipy
from scipy import stats
from typing import Tuple

import opfgym
import opfgym.util
import opfgym.objective
import opfgym.sampling
from opfgym.simbench.data_split import define_test_train_split
from opfgym.simbench.time_observation import get_simbench_time_observation


class PowerFlowNotAvailable(Exception):
    pass


class OpfEnv(gym.Env):
    def __init__(self,
                 net: pp.pandapowerNet,
                 action_keys: tuple[tuple[str, str, np.ndarray], ...],
                 observation_keys: tuple[tuple[str, str, np.ndarray], ...],
                 state_keys: tuple[tuple[str, str, np.ndarray], ...]=None,
                 profiles: dict[str, pd.DataFrame]=None,
                 steps_per_episode: int=1,
                 bus_wise_obs: bool=False,
                 reward_function: opfgym.RewardFunction=None,
                 reward_function_params: dict=None,
                 diff_objective: bool=False,
                 add_res_obs: bool=False,
                 add_time_obs: bool=False,
                 add_act_obs: bool=False,
                 add_mean_obs: bool=False,
                 train_sampling: str='simbench',
                 test_sampling: str='simbench',
                 validation_sampling: str='simbench',
                 evaluate_on: str='validation',
                 sampling_params: dict=None,
                 constraint_params: dict={},
                 custom_constraints: list=None,
                 autoscale_actions: bool=True,
                 diff_action_step_size: float=None,
                 clipped_action_penalty: float=0.0,
                 initial_action: str='center',
                 objective_function: Callable[[pp.pandapowerNet], np.ndarray | float]=None,
                 power_flow_solver: Callable[[pp.pandapowerNet], None]=None,
                 optimal_power_flow_solver: Callable[[pp.pandapowerNet], None]=None,
                 seed: int=None,
                 **kwargs):

        self.net = net
        self.obs_keys = observation_keys
        self.state_keys = state_keys or copy.copy(observation_keys)
        self.act_keys = action_keys
        self.profiles = profiles

        if not profiles:
            assert 'simbench' not in test_sampling
            assert 'simbench' not in train_sampling
            assert 'simbench' not in validation_sampling
            assert not add_time_obs

        # Define data sampling
        self.evaluate_on = evaluate_on
        sampling_params = sampling_params or {}
        split = define_test_train_split(**kwargs)
        self.test_steps, self.validation_steps, self.train_steps = split
        if isinstance(train_sampling, opfgym.sampling.DatasetSampler):
            self.train_sampling = train_sampling
        elif isinstance(train_sampling, str):
            self.train_sampling = opfgym.sampling.create_default_sampler(
                train_sampling, self.state_keys, profiles, self.train_steps,
                seed, **sampling_params)
        if isinstance(validation_sampling, opfgym.sampling.DatasetSampler):
            self.validation_sampling = validation_sampling
        elif isinstance(validation_sampling, str):
            self.validation_sampling = opfgym.sampling.create_default_sampler(
                validation_sampling, self.state_keys, profiles,
                self.validation_steps, seed, **sampling_params)
        if isinstance(test_sampling, opfgym.sampling.DatasetSampler):
            self.test_sampling = test_sampling
        elif isinstance(test_sampling, str):
            self.test_sampling = opfgym.sampling.create_default_sampler(
                test_sampling, self.state_keys, profiles, self.test_steps, seed,
                **sampling_params)
        self.current_sampler = None

        # Define the power flow and OPF solvers (default to pandapower)
        self._run_power_flow = power_flow_solver or self.default_power_flow
        if optimal_power_flow_solver is None:
            self._run_optimal_power_flow = self.default_optimal_power_flow
        elif optimal_power_flow_solver is False:
            # No optimal power flow solver available
            self._run_optimal_power_flow = raise_opf_not_converged
        else:
            self._run_optimal_power_flow = optimal_power_flow_solver

        # Define objective function
        if objective_function is None:
            self.objective_function = opfgym.objective.get_pandapower_costs
        else:
            assert_only_net_in_signature(objective_function)
            self.objective_function = objective_function

        # self.evaluate_on = evaluate_on
        # self.train_sampling = train_sampling
        # self.test_sampling = test_sampling
        # self.sampling_params = sampling_params or {}

        # Define the observation space
        self.add_act_obs = add_act_obs
        if add_act_obs:
            # The agent can observe its previous actions
            self.obs_keys.extend(self.act_keys)

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

        # Define observation, state, and action spaces
        self.bus_wise_obs = bus_wise_obs
        self.observation_space = get_obs_and_state_space(
            self.net, self.obs_keys, add_time_obs, add_mean_obs,
            seed=seed, bus_wise_obs=bus_wise_obs)
        self.state_space = get_obs_and_state_space(
            self.net, self.state_keys, seed=seed)
        n_actions = sum([len(idxs) for _, _, idxs in self.act_keys])
        self.action_space = gym.spaces.Box(0, 1, shape=(n_actions,), seed=seed)

        # Action space details
        self.autoscale_actions = autoscale_actions
        self.diff_action_step_size = diff_action_step_size
        self.clipped_action_penalty = clipped_action_penalty
        self.initial_action = initial_action

        self.steps_per_episode = steps_per_episode

        # Full state of the system (available in training, but not in testing)
        self.state = None  # TODO: Not implemented yet. Required only for partially observable envs

        # Is a powerflow calculation required to get new observations in reset?
        self.pf_for_obs = False
        for unit_type, _, _ in self.obs_keys:
            if 'res_' in unit_type:
                self.pf_for_obs = True
                break

        self.diff_objective = diff_objective
        if diff_objective:
            # An initial power flow is required to compute the initial objective
            self.pf_for_obs = True

        # # Define data distribution for training and testing
        # self.test_steps, self.validation_steps, self.train_steps = define_test_train_split(**kwargs)

        # Constraints
        if custom_constraints is None:
            self.constraints = opfgym.constraints.create_default_constraints(
                self.net, constraint_params)
        else:
            self.constraints = custom_constraints

        # Define reward function
        reward_function_params = reward_function_params or {}
        if reward_function is None:
            # Default reward
            self.reward_function = opfgym.reward.Summation(
                env=self, **reward_function_params)
        elif isinstance(reward_function, str):
            # Load by string (e.g. 'Summation' or 'summation')
            reward_class = opfgym.util.load_class_from_module(
                reward_function, 'opfgym.reward')
            self.reward_function = reward_class(
                env=self, **reward_function_params)
        elif isinstance(reward_function, opfgym.RewardFunction):
            # User-defined reward function
            self.reward_function = reward_function

    def reset(self, seed: int=None, options: dict=None) -> tuple[np.ndarray, dict]:
        """ gymnasium API. Reset the environment to a new state. Samples a 
        random state from the given data distribution, applies an initial action,
        runs a power flow calculation (optional), and returns the initial
        observation.

        :param seed: Seed for the random number generator.
        :param options: Additional options for the reset method. 
            Available options: 'step' (int) to control the data sampling, 
            'test' (bool) to sample from test data."""
        super().reset(seed=seed)
        self.info = {}
        self.step_in_episode = 0

        if not options:
            options = {}

        test = options.get('test', False)
        step = options.get('step', None)
        # self.apply_action = options.get('new_action', True)

        self._sampling(step, test, seed)

        if self.initial_action == 'random':
            # Use random actions as starting point so that agent learns to handle that
            act = self.action_space.sample()
        else:
            # Reset all actions to default values
            act = (self.action_space.low + self.action_space.high) / 2
        self._apply_actions(act)

        if self.pf_for_obs is True:
            self.run_power_flow()
            if not self.power_flow_available:
                logging.warning(
                    'Failed powerflow calculcation in reset. Try again!')
                return self.reset()

            self.initial_obj = self.calculate_objective(diff_objective=False)

        obs = self._get_obs(self.obs_keys, self.add_time_obs, self.add_mean_obs)

        return obs, copy.deepcopy(self.info)

    def _sampling(self, step=None, test=False, seed=None, **kwargs) -> None:

        self.set_power_flow_unavailable()

        if not test:
            self.current_sampler = self.train_sampling
        elif self.evaluate_on == 'validation':
            self.current_sampler = self.validation_sampling
        elif self.evaluate_on == 'test':
            self.current_sampler = self.test_sampling

        if seed:
            self.current_sampler.set_seed(seed)

        self.net = self.current_sampler(self.net, step, **kwargs)

    @property
    def current_time_step(self):
        try:
            return self.current_sampler.current_time_step
        except AttributeError:
            return None

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """ gymnasium API: Step the environment to a new state. 
        Applies the actions, runs a power flow, checks for constraint
        violations, calculates the reward, and returns all information requires
        for learning

        :param action: The action to apply to the power system.
        """
        assert not np.isnan(action).any()
        self.info = {}
        self.step_in_episode += 1

        # if self.apply_action:
        correction = self._apply_actions(action, self.diff_action_step_size)
        self.run_power_flow()

        if not self.power_flow_available:
            # Something went seriously wrong! Find out what!
            # Maybe NAN in power setpoints?!
            # Maybe simply catch this with a strong negative reward?!
            logging.critical(f'\nPowerflow not converged and reason unknown! Run diagnostic tool to at least find out what went wrong: {pp.diagnostic(self.net)}')
            self.info['valids'] = np.array([False] * 5)
            self.info['violations'] = np.array([1] * 5)
            self.info['unscaled_penalties'] = np.array([1] * 5)
            self.info['penalty'] = 5
            return np.array([np.nan]), np.nan, True, False, copy.deepcopy(self.info)

        reward = self.calculate_reward()

        if self.clipped_action_penalty: #  and self.apply_action:
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

        obs = self._get_obs(self.obs_keys, self.add_time_obs, self.add_mean_obs)
        assert not np.isnan(obs).any()

        return obs, reward, terminated, truncated, copy.deepcopy(self.info)

    def _apply_actions(self, action, diff_action_step_size=None) -> float:
        """ Apply agent actions as setpoints to the power system at hand.
        Returns the mean correction that was necessary to make the actions
        valid."""

        self.set_power_flow_unavailable()

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

            self.net[unit_type].loc[idxs, actuator] = setpoints

            counter += len(idxs)

        # Did the action need to be corrected to be in bounds?
        mean_correction = np.mean(abs(
            self.get_current_actions(from_results_table=False) - action))

        return mean_correction

    def calculate_objective(self, net=None, diff_objective=False) -> np.ndarray:
        """ This method returns the objective function as array that is used as
        basis for the reward calculation. """
        net = net or self.net
        if diff_objective:
            return -self.objective_function(net) - self.initial_obj
        else:
            return -self.objective_function(net)

    def calculate_violations(self, net=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        net = net or self.net
        valids = []
        violations = []
        penalties = []
        for constraint in self.constraints:
            result = constraint.get_violation_metrics(net)
            valids.append(result['valid'])
            violations.append(result['violation'])
            penalties.append(result['penalty'])

        return np.array(valids), np.array(violations), np.array(penalties)

    def calculate_reward(self) -> float:
        """ Combine objective function and the penalties together. """
        objective = np.sum(self.calculate_objective(diff_objective=self.diff_objective))
        valids, violations, penalties = self.calculate_violations()

        self.info['valids'] = np.array(valids)
        self.info['violations'] = np.array(violations)
        self.info['unscaled_penalties'] = np.array(penalties)

        penalty = np.sum(penalties)
        valid = valids.all()

        reward = self.reward_function(objective, penalty, valid)
        self.info['cost'] = self.reward_function.calculate_cost(penalty, valid)

        return reward

    def _get_obs(self, obs_keys, add_time_obs=False, add_mean_obs=False
                 ) -> np.ndarray:
        obss = [(self.net[unit_type].loc[idxs, column].to_numpy())
                if (unit_type != 'load' or not self.bus_wise_obs)
                else get_bus_aggregated_obs(self.net, 'load', column, idxs)
                for unit_type, column, idxs in obs_keys]

        if add_mean_obs:
            mean_obs = [np.mean(partial_obs) for partial_obs in obss 
                        if len(partial_obs) > 1]
            obss.append(mean_obs)

        if add_time_obs and self.current_time_step is not None:
            time_obs = get_simbench_time_observation(
                self.profiles, self.current_time_step)
            obss = [time_obs] + obss

        return np.concatenate(obss)

    def get_state(self) -> np.ndarray:
        """ Return the state of the underlying power system. Relevant for
        partially observable environments. Steal the popgym API for this, compare
        https://popgym.readthedocs.io/en/latest/autoapi/popgym/core/env/index.html
        """
        return self._get_obs(self.state_keys)

    def render(self, **kwargs):
        """ gymnasium API. Render the current state of the power system. Uses the `simple_plot`
        pandapower method. Overwrite for more sophisticated rendering. For
        kwargs information, refer to the pandapower docs:
        https://pandapower.readthedocs.io/en/latest/plotting/matplotlib/simple_plot.html"""
        ax = pp.plotting.simple_plot(self.net, **kwargs)
        return ax

    def get_current_actions(self, net=None, from_results_table=True) -> np.ndarray:
        # Attention: These are not necessarily the actions of the RL agent
        # because some re-scaling might have happened!
        # These are the actions from the original action space [0, 1]
        net = net or self.net
        res_prefix = 'res_' if from_results_table else ''
        action = []
        for unit_type, column, idxs in self.act_keys:
            setpoints = net[f'{res_prefix}{unit_type}'].loc[idxs, column]

            # If data not taken from results table, scaling required
            if not from_results_table and 'scaling' in net[unit_type].columns:
                setpoints *= net[unit_type].scaling.loc[idxs]

            # Action space depends on autoscaling 
            min_id = 'min_' if self.autoscale_actions else 'min_min_'
            max_id = 'max_' if self.autoscale_actions else 'max_max_' 
            min_values = net[unit_type][f'{min_id}{column}'].loc[idxs]
            max_values = net[unit_type][f'{max_id}{column}'].loc[idxs]

            action.append((setpoints - min_values) / (max_values - min_values))

        return np.concatenate(action)

    def get_actions(self) -> np.ndarray:
        """ Returns the current actions that were applied to the power system.

        Warning: Not necessarily the exact same actions that were used
        in the :meth:`step` method because some rounding, clipping, etc. might
        have happened. However, the resulting power flows should be same.
        Useful for storing and reproducing results.
        """
        if self.power_flow_available:
            return self.get_current_actions(from_results_table=True)
        return self.get_current_actions(from_results_table=False)

    def get_optimal_actions(self) -> np.ndarray:
        """ Returns the optimal actions that were calculated by the OPF.
        Useful for creating datasets for supervised learning.

        Warning: Can only be called if :meth:`run_optimal_power_flow` method was
        called before.
        """
        self.ensure_optimal_power_flow_available()
        # The pandapower OPF stores the optimal settings only in the results table
        return self.get_current_actions(self.optimal_net, from_results_table=True)

    def is_state_valid(self) -> bool:
        """ Returns True if the current state does not contain constraint
        violations. """
        self.ensure_power_flow_available()
        valids, _, _ = self.calculate_violations(self.net)
        return valids.all()

    def is_optimal_state_valid(self) -> bool:
        """ Returns True if the state after OPF calculation does not contain
        constraint violations.

        Warning: Can only be called if :meth:`run_optimal_power_flow` method was called before. 

        Warning 2: Usually, the OPF does not converge if no valid solutions
        can be found. This method is only applicable if the OPF yielded a
        solution. However, a non-converged OPF can be counted as an invalid
        state in most cases.
        """
        self.ensure_optimal_power_flow_available()
        valids, _, _ = self.calculate_violations(self.optimal_net)
        return valids.all()

    def get_objective(self) -> float:
        """ Returns the currrent value of the objective function. """
        self.ensure_power_flow_available()
        return sum(self.calculate_objective(self.net))

    def get_optimal_objective(self) -> float:
        """ Returns the optimal value of the objective function. Warning: Can
        only be called if :meth:`run_optimal_power_flow` method was called before. """
        self.ensure_optimal_power_flow_available()
        return sum(self.calculate_objective(self.optimal_net))

    def run_power_flow(self, **kwargs):
        """ Updates the current power system state with
        the respective power flow (line loading, voltage magnitudes, etc.).
        Should be called whenever the power system state changed. The keyword
        arguments can be used to pass additional arguments to the
        power flow solver.

        :param kwargs: Additional arguments for the power flow solver. 
            (default: pandapower. Compare: https://pandapower.readthedocs.io/en/latest/powerflow/ac.html)
        """
        try:
            self._run_power_flow(self.net, **kwargs)
            self.power_flow_available = True
            return True
        except pp.powerflow.LoadflowNotConverged:
            logging.warning('Powerflow not converged!!!')
            return False

    def run_optimal_power_flow(self, **kwargs):
        """ Creates and internal copy of the power system with its current state
        and performs the OPF on that copy. Should be called to compare the current solution
        with the optimal solution. The keyword arguments can be used to pass additional
        arguments to the pandapower OPF solver.

        :param kwargs: Additional arguments for the OPF solver.
            (default: pandapower. Compare: https://pandapower.readthedocs.io/en/latest/opf/formulation.html)
        """
        self.optimal_net = copy.deepcopy(self.net)
        try:
            self._run_optimal_power_flow(self.optimal_net, **kwargs)
            self.optimal_power_flow_available = True
            return True
        except pp.optimal_powerflow.OPFNotConverged:
            logging.warning('OPF not converged!!!')
            return False

    def ensure_power_flow_available(self):
        if not self.power_flow_available:
            raise PowerFlowNotAvailable('Please call `run_power_flow` first!')

    def ensure_optimal_power_flow_available(self):
        if not self.optimal_power_flow_available:
            raise PowerFlowNotAvailable('Please call `run_optimal_power_flow` first!')

    def set_power_flow_unavailable(self):
        """ Reset the power flow availability to indicate that a new power flow
        or OPF calculation is required. """
        self.power_flow_available = False
        self.optimal_power_flow_available = False

    @staticmethod
    def default_power_flow(net, enforce_q_lims=True, **kwargs):
        """ Default power flow: Use the pandapower power flow.

        Default setting: Enforce q limits as automatic constraint satisfaction.
        """
        pp.runpp(net, enforce_q_lims=enforce_q_lims, **kwargs)

    @staticmethod
    def default_optimal_power_flow(net, calculate_voltage_angles=False, **kwargs):
        """ Default OPF: Use the pandapower OPF.

        Default setting: Do not calculate voltage angles because often results
        in errors for SimBench nets. """
        pp.runopp(net, calculate_voltage_angles=calculate_voltage_angles, **kwargs)


def get_obs_and_state_space(net: pp.pandapowerNet, obs_or_state_keys: list,
                            add_time_obs: bool=False, add_mean_obs: bool=False,
                            seed: int=None, last_n_obs: int=1,
                            bus_wise_obs=False) -> gym.spaces.Box:
    """ Get observation or state space from the constraints of the power
    network. """
    lows, highs = [], []

    if add_time_obs:
        # Time is always given as observation of lenght 6 in range [-1, 1]
        # at the beginning of the observation!
        lows.append(-np.ones(6))
        highs.append(np.ones(6))

    for unit_type, column, idxs in obs_or_state_keys:
        if 'res_' in unit_type:
            # The constraints are never defined in the results table
            unit_type = unit_type[4:]
        elif 'max_' in column or 'min_' in column:
            # If the constraint itself is an observation, treat is the same as a normal observation -> remove prefix
            column = column[4:]

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
            if len(l) > 0 and len(l) == len(h):
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


def assert_only_net_in_signature(function):
    """ Ensure that the function only takes a pandapower net as argument. """
    signature = inspect.signature(function)
    message = 'Function must only take a pandapower net as argument!'
    assert list(signature.parameters.keys()) == ['net'], message


def raise_opf_not_converged(net, **kwargs):
    raise pp.optimal_powerflow.OPFNotConverged(
        "OPF solver not available for this environment.")
