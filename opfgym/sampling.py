

import abc

import numpy as np
import pandapower as pp
import scipy
from scipy import stats


class DatasetSampler(abc.ABC):
    def __init__(self, seed=None, **kwargs) -> None:
        self.np_random = np.random.RandomState(seed=seed)

    def __call__(self, net, step: int=None, *args, **kwargs):
        self.sample_state(net, step=step, *args, **kwargs)
        return net

    @abc.abstractmethod
    def sample_state(
        self, net: pp.pandapowerNet, *args, **kwargs
            ) -> pp.pandapowerNet:
        pass

    def set_seed(self, seed):
        self.np_random.seed(seed)


class SequentialSampler(DatasetSampler):
    """ Combines multiple samplers to one sampler by calling them sequentially.
    Should be used to combine different sampling strategies, for example,
    by sampling load data from simbench and price data with uniform distribution
    in some range (price data is not available for simbench).

    Args:
        samplers: Tuple of samplers to be combined.

    """
    def __init__(self, samplers: tuple, seed=None, **kwargs) -> None:
        super().__init__(seed=seed, **kwargs)
        self.samplers = samplers

        # Make sure that all samplers have the same starting seed
        self.set_seed(seed)

    def set_seed(self, seed):
        super().set_seed(seed)
        for sampler in self.samplers:
            sampler.set_seed(seed)

    def sample_state(self, net, *args, **kwargs):
        for sampler in self.samplers:
            net = sampler.sample_state(net, *args, **kwargs)
        return net

    def __len__(self):
        return len(self.samplers)

    def __getitem__(self, idx: int):
        return self.samplers[idx]

    def __getattr__(self, attr: str):
        for sampler in self.samplers:
            try: 
                return getattr(sampler, attr)
            except AttributeError:
                pass
        raise AttributeError(f'None of the samplers has the attribute {attr}.')


class SimbenchSampler(DatasetSampler):
    def __init__(self,
                 state_keys: tuple,
                 profiles: dict,
                 available_steps: np.ndarray=None,
                 in_between_steps=False,
                 noise_factor=0.0,
                 noise_distribution='uniform',
                 **kwargs) -> None:
        self.state_keys = state_keys
        self.profiles = profiles
        self.in_between_steps = in_between_steps
        self.noise_factor = noise_factor
        self.noise_distribution = noise_distribution
        self.total_n_steps = tuple(profiles.values())[0].shape[0]

        if available_steps is None:
            self.available_steps = np.arange(self.total_n_steps)
        else:
            self.available_steps = np.array(available_steps)

        super().__init__(**kwargs)

    def sample_state(self, net, step: int=None, **kwargs) -> pp.pandapowerNet:
        """ Standard pre-implemented method to sample a random state from the
        simbench time-series data and set that state.

        Works only for simbench systems!
        """

        if step is None:
            step = self.np_random.choice(self.available_steps)
        else:
            assert step < self.total_n_steps

        self.current_time_step = step

        for unit_type, column, idxs in self.state_keys:
            type_act = (unit_type, column)
            if len(idxs) == 0 or not self.profiles[type_act].shape[1]:
                continue
            data = self.profiles[type_act].loc[step, idxs]

            if self.in_between_steps and step < self.total_n_steps - 1:
                # Random linear interpolation between two steps
                next_data = self.profiles[type_act].loc[step + 1, idxs]
                r = self.np_random.random()
                data = data * r + next_data * (1 - r)

            # Add some noise to create unique data samples
            if self.noise_distribution == 'uniform':
                # Uniform distribution: noise_factor as relative sample range
                noise = self.np_random.random(
                    len(idxs)) * self.noise_factor * 2 + (1 - self.noise_factor)
                new_values = (data * noise)
            elif self.noise_distribution == 'normal':
                # Normal distribution: noise_factor as relative std deviation
                new_values = self.np_random.normal(
                    loc=data, scale=data.abs() * self.noise_factor)

            # Make sure that the range of original data remains unchanged
            # (Technical limits of the units remain the same)
            new_values = new_values.clip(
                self.profiles[type_act].min()[idxs],
                self.profiles[type_act].max()[idxs]
            )

            net[unit_type].loc[idxs, column] = new_values

        return net


class NormalSampler(DatasetSampler):
    def __init__(self, 
                 state_keys: tuple, 
                 relative_standard_deviation: float=None, 
                 truncated: bool=False, 
                 **kwargs
            ) -> None:
        self.state_keys = state_keys
        self.relative_standard_deviation = relative_standard_deviation
        self.truncated = truncated
        super().__init__(**kwargs)

    def sample_state(self, net: pp.pandapowerNet, **kwargs) -> pp.pandapowerNet:
        """ Sample data around mean values from simbench data. """
        for unit_type, column, idxs in self.state_keys:
            if 'res_' in unit_type:
                continue

            df = net[unit_type].loc[idxs]
            mean = df[f'mean_{column}']

            # TODO: This piece of code is used multiple times -> use fucntion?!
            try:
                max_values = (df[f'max_max_{column}'] / df.scaling).to_numpy()
            except KeyError:
                max_values = (df[f'max_{column}'] / df.scaling).to_numpy()
            try:
                min_values = (df[f'min_min_{column}'] / df.scaling).to_numpy()
            except KeyError:
                min_values = (df[f'min_{column}'] / df.scaling).to_numpy()

            diff = max_values - min_values

            if self.relative_standard_deviation:
                std = self.relative_standard_deviation * diff
            else:
                std = df[f'std_dev_{column}']

            if self.truncated:
                # Make sure to re-distribute truncated values
                random_values = stats.truncnorm.rvs(
                    min_values, max_values, mean, std * diff, len(mean))
            else:
                # Simply clip values to min/max range
                random_values = self.np_random.normal(
                    mean, std * diff, len(mean))
                random_values = np.clip(
                    random_values, min_values, max_values)
            net[unit_type][column].loc[idxs] = random_values

        return net


class UniformSampler(DatasetSampler):
    def __init__(self, state_keys: tuple, **kwargs) -> None:
        self.state_keys = state_keys
        super().__init__(**kwargs)

    def sample_state(self, net: pp.pandapowerNet, **kwargs) -> pp.pandapowerNet:
        """ Standard pre-implemented method to set power system to a new random
        state from uniform sampling. Uses the observation space as basis.
        Requirement: For every observations there must be "min_{obs}" and
        "max_{obs}" given as range to sample from.
        """
        for unit_type, column, idxs in self.state_keys:
            # Results cannot be sampled but only be computed by the power flow
            if 'res_' not in unit_type:
                self._sample_from_range(net, unit_type, column, idxs)

        return net

    def _sample_from_range(self, net, unit_type:str, column: str, idxs: np.ndarray) -> None:
        df = net[unit_type]
        # Make sure to sample from biggest possible range
        try:
            low = df[f'min_min_{column}'].loc[idxs]
        except KeyError:
            low = df[f'min_{column}'].loc[idxs]
        try:
            high = df[f'max_max_{column}'].loc[idxs]
        except KeyError:
            high = df[f'max_{column}'].loc[idxs]

        r = self.np_random.uniform(low, high, size=(len(idxs),))
        try:
            # Range definitions are scaled, which is why we need to unscale
            net[unit_type][column].loc[idxs] = r / df.scaling[idxs]
        except AttributeError:
            # If scaling factor is not defined, scaling=1
            net[unit_type][column].loc[idxs] = r


class MixedRandomSampler(DatasetSampler):
    """ Combines multiple samplers to one sampler by calling one of them
    randomly. For example, can be used to sample either from simbench data or
    from random data to create a more diverse dataset.

    Args:
        samplers: Tuple of samplers to be combined.
        sampler_probabilities_cumulated: Tuple of probabilities to sample from
            the corresponding sampler. Must be in ascending order. For example,
            (0.4, 1.0) to sample from the first sampler with 40% probability
            and from the second sampler with 60% probability.
    """
    def __init__(self,
                 samplers: tuple,
                 sampler_probabilities_cumulated: tuple,
                 seed=None,
                 **kwargs) -> None:
        self.samplers = samplers
        self.sampler_probabilities_cumulated = sampler_probabilities_cumulated
        assert len(samplers) == len(sampler_probabilities_cumulated)
        assert sampler_probabilities_cumulated[-1] == 1
        super().__init__(seed=seed, **kwargs)
        self.set_seed(seed)

    def sample_state(self, net: pp.pandapowerNet, *args, **kwargs):
        random_number = self.np_random.random()
        for idx, sampler in enumerate(self.samplers):
            if random_number < self.sampler_probabilities_cumulated[idx]:
                return sampler.sample_state(net, *args, **kwargs)

    def set_seed(self, seed):
        super().set_seed(seed)
        for sampler in self.samplers:
            sampler.set_seed(seed)


class StandardMixedRandomSampler(MixedRandomSampler):
    """ Standard combination of sampling from either simbench data, uniform,
    distributed data or normal distributed data (combination of the three 
    standard samplers)."""
    def __init__(self, **kwargs) -> None:
        simbench_sampler = SimbenchSampler(**kwargs)
        normal_sampler = NormalSampler(**kwargs)
        uniform_sampler = UniformSampler(**kwargs)
        samplers = (simbench_sampler, normal_sampler, uniform_sampler)
        super().__init__(samplers, **kwargs)


def create_default_sampler(sampler: str,
                           state_keys: tuple,
                           profiles: dict,
                           available_steps: np.ndarray=None,
                           seed=None,
                           **kwargs) -> DatasetSampler:
    """ Default sampler: Always use uniform sampling for prices and one of
    'simbench', 'full_uniform', or 'normal_around_mean' distribution sampling for the rest.

    :param state_keys: Keys to sample from the state space.
    TODO: Add params for all this
    """

    # Simbench provides only time-series data for power values (e.g. no costs)
    simbench_condition = lambda key: 'p_mw' in key[1] or 'q_mvar' in key[1]

    # Use uniform distribution for everything else by default
    uniform_keys = [key for key in state_keys if not simbench_condition(key)]
    uniform_sampler = UniformSampler(uniform_keys, seed=seed, **kwargs)

    rest_keys = [key for key in state_keys if simbench_condition(key)]
    if sampler == 'simbench':
        user_sampler = SimbenchSampler(rest_keys,
                                       profiles=profiles,
                                       available_steps=available_steps,
                                       **kwargs)
    elif sampler == 'normal_around_mean':
        user_sampler = NormalSampler(rest_keys, **kwargs)
    elif sampler == 'full_uniform':
        user_sampler = UniformSampler(rest_keys, **kwargs)
    elif sampler == 'mixed':
        user_sampler = StandardMixedRandomSampler(rest_keys, **kwargs)
    else:
        raise ValueError(f"Sampler {sampler} not availabe in opfgym.")

    return SequentialSampler((user_sampler, uniform_sampler), seed=seed)
