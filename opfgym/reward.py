
import abc

import numpy as np


class RewardFunction(abc.ABC):
    def __init__(self,
                 penalty_weight: float = 0.5,
                 clip_range: tuple[float, float] = None,
                 reward_scaling: str = None,
                 reward_scaling_params: dict = None,
                 env = None):
        self.penalty_weight = penalty_weight
        self.clip_range = clip_range

        self.prepare_reward_scaling(reward_scaling, reward_scaling_params, env)

    def prepare_reward_scaling(self,
                               reward_scaling: str,
                               reward_scaling_params: dict,
                               env) -> None:
        """ Prepare the reward scaling parameters for later use. """
        reward_scaling_params = reward_scaling_params or {}
        if reward_scaling_params == 'auto' or (
                'num_samples' in reward_scaling_params) or (
                not reward_scaling_params and reward_scaling):
            scaling_params = estimate_reward_distribution(env, **reward_scaling_params)
        else:
            scaling_params = reward_scaling_params

        if not reward_scaling:
            scaling_params.update({'penalty_factor': 1, 'penalty_bias': 0,
                                   'objective_factor': 1, 'objective_bias': 0})
        elif reward_scaling == 'minmax11':
            scaling_params.update(calculate_minmax_11_params(**scaling_params))
        elif reward_scaling == 'minmax01':
            scaling_params.update(calculate_minmax_01_params(**scaling_params))
        elif reward_scaling == 'normalization':
            scaling_params.update(calculate_normalization_params(**scaling_params))
        else:
            raise NotImplementedError('This reward scaling does not exist!')

        # Error handling if there were no constraint violations
        if np.isnan(scaling_params['penalty_bias']):
            scaling_params['penalty_bias'] = 0
        if np.isinf(scaling_params['penalty_factor']):
            scaling_params['penalty_factor'] = 1

        self.scaling_params = scaling_params

    def __call__(self, objective: float, penalty: float, valid: bool) -> float:
        objective = self.adjust_objective(objective, valid)
        penalty = self.adjust_penalty(penalty, valid)

        objective = self.scale_objective(objective)
        penalty = self.scale_penalty(penalty)

        reward = self.compute_total_reward(objective, penalty)

        if self.clip_range:
            reward = self.clip_reward(reward)

        return reward

    def clip_reward(self, reward: float) -> float:
        return np.clip(reward, self.clip_range[0], self.clip_range[1])

    def compute_total_reward(self, objective: float, penalty: float) -> float:
        if self.penalty_weight is None:
            return objective + penalty
        return objective * (1 - self.penalty_weight) + penalty * self.penalty_weight

    def scale_objective(self, objective: float) -> float:
        objective *= self.scaling_params['objective_factor']
        objective += self.scaling_params['objective_bias']
        return objective

    def scale_penalty(self, penalty: float) -> float:
        penalty *= self.scaling_params['penalty_factor']
        penalty += self.scaling_params['penalty_bias']
        return penalty

    def calculate_cost(self, penalty, valid) -> float:
        """ For safe RL algorithms, we need to compute a cost value > 0.
        Therefore, it is important to prevent any sign change. """
        if valid:
            return 0.0
        return abs(penalty * self.scaling_params['penalty_factor'])

    @abc.abstractmethod
    def adjust_penalty(self, penalty: float, valid: bool) -> float:
        return penalty

    @abc.abstractmethod
    def adjust_objective(self, objective: float, valid: bool) -> float:
        return objective


def calculate_normalization_params(
        std_objective: float,
        mean_objective: float,
        std_penalty: float,
        mean_penalty: float,
        **kwargs
        ) -> dict:
    """
    Scale so that mean is zero and standard deviation is one
    formula: (obj - mean_objective) / obj_std
    """
    params = {}
    params['objective_factor'] = 1 / std_objective
    params['objective_bias'] = -mean_objective / std_objective
    params['penalty_factor'] = 1 / std_penalty
    params['penalty_bias'] = -mean_penalty / std_penalty
    return params


def calculate_minmax_01_params(
        min_objective: float,
        max_objective: float,
        min_penalty: float,
        max_penalty: float,
        **kwargs
        ) -> dict:
    """
    Scale from range [min, max] to range [0, 1].
    formula: (obj - min_objective) / (max_objective - min_objective)
    """
    params = {}
    diff = (max_objective - min_objective)
    params['objective_factor'] = 1 / diff
    params['objective_bias'] = -(min_objective / diff)
    diff = (max_penalty - min_penalty)
    params['penalty_factor'] = 1 / diff
    params['penalty_bias'] = -(min_penalty / diff)
    return params


def calculate_minmax_11_params(
        min_objective: float,
        max_objective: float,
        min_penalty: float,
        max_penalty: float,
        **kwargs
        ) -> dict:
    """
    Scale from range [min, max] to range [-1, 1].
    formula: (obj - min_objective) / (max_objective - min_objective) * 2 - 1
    """
    params = {}
    diff = (max_objective - min_objective) / 2
    params['objective_factor'] = 1 / diff
    params['objective_bias'] = -(min_objective / diff + 1)
    diff = (max_penalty - min_penalty) / 2
    params['penalty_factor'] = 1 / diff
    params['penalty_bias'] = -(min_penalty / diff + 1)
    return params


def estimate_reward_distribution(env, num_samples: int=3000) -> dict:
    """ Get normalization parameters for scaling down the reward. """
    objectives = []
    penalties = []
    for _ in range(num_samples):
        # Apply random actions to random states
        env.reset()
        # Use _apply_actions() to ensure that the action space definition is kept outside (in contrast to step())
        env._apply_actions(env.action_space.sample())
        env.run_power_flow()
        objectives.append(env.calculate_objective(env.net))
        penalties.append(env.calculate_violations()[2])

    objectives = np.array(objectives).sum(axis=1)
    penalties = np.array(penalties).sum(axis=1)

    # Remove potential NaNs (due to failed power flows or similar)
    objectives = objectives[~np.isnan(objectives)]
    penalties = penalties[~np.isnan(penalties)]

    norm_params = {
        'min_objective': objectives.min(),
        'max_objective': objectives.max(),
        'min_penalty': penalties.min(),
        'max_penalty': penalties.max(),
        'mean_objective': objectives.mean(),
        'mean_penalty': penalties.mean(),
        'std_objective': np.std(objectives),
        'std_penalty': np.std(penalties),
        'median_objective': np.median(objectives),
        'median_penalty': np.median(penalties),
        'mean_abs_objective': np.abs(objectives).mean(),
        'mean_abs_penalty': np.abs(penalties).mean(),
    }

    return norm_params


class Summation(RewardFunction):
    """ Simply add up the objective and penalty to create the reward.
    Often used in literature, compare
    https://www.sciencedirect.com/science/article/pii/S2666546824000764"""
    def adjust_penalty(self, penalty, valid) -> float:
        return penalty

    def adjust_objective(self, objective, valid) -> float:
        return objective


class Replacement(RewardFunction):
    """ Return either objective or penalty, depending on the validity of the
    solution. Often used in literature, compare
    https://www.sciencedirect.com/science/article/pii/S2666546824000764"""
    def __init__(self, valid_reward: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        if isinstance(valid_reward, str):
            self.valid_reward = get_valid_reward_from_heuristic(
                valid_reward, self.scaling_params)
        else:
            self.valid_reward = valid_reward

    def adjust_penalty(self, penalty, valid) -> float:
        return penalty

    def adjust_objective(self, objective, valid) -> float:
        """ Only allow reward for optimization if solution is valid."""
        if valid:
            # Make sure that the valid reward is always higher than invalid one
            return objective + self.valid_reward
        return 0.0


class Parameterized(RewardFunction):
    """ Combination of the summation and replacement reward. Allows all
    intermediates in between the two to find the best combination.

    Example:
    If valid_reward==0 & invalid_objective_share==1: Summation reward
    If valid_reward>0 & invalid_objective_share==0: Replacement reward

    The range in between represents weighted combinations of both
    The invalid_penalty is added to allow for inverse replacement method
    """
    def __init__(self,
                 valid_reward: float = 0.0,
                 invalid_penalty: float = 0.5,
                 invalid_objective_share: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(valid_reward, str):
            self.valid_reward = get_reward_offset_from_heuristic(
                valid_reward, self.scaling_params)
        else:
            assert valid_reward >= 0, 'Valid reward must be >= 0'
            self.valid_reward = valid_reward

        if isinstance(invalid_penalty, str):
            self.invalid_penalty = get_reward_offset_from_heuristic(
                invalid_penalty, self.scaling_params)
        else:
            assert invalid_penalty >= 0, 'Invalid penalty must be >= 0'
            self.invalid_penalty = invalid_penalty

        assert 0 <= invalid_objective_share <= 1, 'Objective share must be in [0, 1]'
        self.invalid_objective_share = invalid_objective_share

    def adjust_penalty(self, penalty, valid) -> float:
        if valid:
            return penalty + self.valid_reward
        return penalty - self.invalid_penalty

    def adjust_objective(self, objective, valid) -> float:
        if not valid:
            # Make objective part of the reward function smaller to encourage
            # constraint satisfaction first
            objective *= self.invalid_objective_share
        return objective

    def calculate_cost(self, penalty, valid) -> float:
        """ Overwrite base class method to consider invalid_penalty. """
        if valid:
            return 0.0
        return super().calculate_cost(penalty, valid) + self.invalid_penalty


class OnlyObjective(RewardFunction):
    """ Only take the objective into account, ignore the penalty. For example,
    useful for Safe RL algorithms, where the reward is not responsible for
    enforcing constraints, which means that the penalty part of the reward 
    should zero. """
    def __init__(self, **kwargs):
        super().__init__(penalty_weight=0.0, **kwargs)

    def adjust_penalty(self, penalty, valid) -> float:
        return 0.0

    def adjust_objective(self, objective, valid) -> float:
        return objective


def get_reward_offset_from_heuristic(variant: str, scaling_params: dict) -> float:
    """ Heuristic for the valid reward (especially relevant without scaling).
    Compare https://www.sciencedirect.com/science/article/pii/S2666546824000764
    """
    if offset == 'worst':
        offset = scaling_params['min_obj']
    elif offset == 'mean':
        offset = scaling_params['mean_obj']
    offset *= scaling_params['objective_factor']
    offset += scaling_params['objective_bias']
    return offset