

import abc

import numpy as np
import pandapower as pp
import scipy
from scipy import stats


class DatasetSampler(abc.ABC):
    def __init__(self, seed=None, after_sampling_hooks: list=None, **kwargs) -> None:
        self.np_random = np.random.RandomState(seed=seed)
        self.after_sampling_hooks = after_sampling_hooks if after_sampling_hooks else []

    def __call__(self, net, *args, **kwargs):
        self.sample_state(net, *args, **kwargs)
        self._call_hooks(net, *args, **kwargs)
        return net

    @abc.abstractmethod
    def sample_state(self, net, *args, **kwargs) -> pp.pandapowerNet:
        pass

    def _call_hooks(self, net, *args, **kwargs):
        for hook in self.after_sampling_hooks:
            hook(net, *args, **kwargs)
        return net

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
    def __init__(self, samplers: tuple, remove_subsampler_hooks: bool=False,
                 seed=None, **kwargs) -> None:
        super().__init__(seed=seed, **kwargs)
        self.samplers = samplers

        # Sometimes it's useful to remove the hooks to prevent multiple calls
        if remove_subsampler_hooks:
            self._remove_subsampler_hooks()

        # Make sure that all samplers set the same seed
        self.set_seed(seed)

    def _remove_subsampler_hooks(self):
        for sampler in self.samplers:
            sampler.after_sampling_hooks = []

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

    def __getitem__(self, idx):
        return self.samplers[idx]


class SimbenchSampler(DatasetSampler):
    def __init__(self,
                 sample_keys: tuple,
                 profiles: dict,
                 available_steps: np.ndarray=None,
                 in_between_steps=False,
                 noise_factor=0.0,
                 noise_distribution='uniform',
                 **kwargs) -> None:
        self.sample_keys = sample_keys
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

        # self.current_simbench_step = step

        for type_act in self.profiles.keys():
            if not self.profiles[type_act].shape[1]:
                continue
            unit_type, actuator = type_act
            data = self.profiles[type_act].loc[step, net[unit_type].index]

            if self.in_between_steps and step < self.total_n_steps - 1:
                # Random linear interpolation between two steps
                next_data = self.profiles[type_act].loc[step + 1, net[unit_type].index]
                r = self.np_random.random()
                data = data * r + next_data * (1 - r)

            # Add some noise to create unique data samples
            if self.noise_distribution == 'uniform':
                # Uniform distribution: noise_factor as relative sample range
                noise = self.np_random.random(
                    len(net[unit_type].index)) * self.noise_factor * 2 + (1 - self.noise_factor)
                new_values = (data * noise).to_numpy()
            elif self.noise_distribution == 'normal':
                # Normal distribution: noise_factor as relative std deviation
                new_values = self.np_random.normal(
                    loc=data, scale=data.abs() * self.noise_factor)

            # Make sure that the range of original data remains unchanged
            # (Technical limits of the units remain the same)
            new_values = np.clip(
                new_values,
                self.profiles[type_act].min(
                )[net[unit_type].index].to_numpy(),
                self.profiles[type_act].max(
                )[net[unit_type].index].to_numpy())

            net[unit_type].loc[net[unit_type].index, actuator] = new_values

        return net


class NormalSampler(DatasetSampler):
    def __init__(self, sample_keys: tuple, relative_standard_deviation: float=None, truncated: bool=False, **kwargs) -> None:
        self.sample_keys = sample_keys
        self.relative_standard_deviation = relative_standard_deviation
        self.truncated = truncated
        super().__init__(**kwargs)

    def sample_state(self, net, **kwargs) -> pp.pandapowerNet:
        """ Sample data around mean values from simbench data. """
        for unit_type, column, idxs in self.sample_keys:
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

            print(min_values, max_values)

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
    def __init__(self, sample_keys: tuple, **kwargs) -> None:
        self.sample_keys = sample_keys
        super().__init__(**kwargs)

    def sample_state(self, net, **kwargs) -> pp.pandapowerNet:
        """ Standard pre-implemented method to set power system to a new random
        state from uniform sampling. Uses the observation space as basis.
        Requirement: For every observations there must be "min_{obs}" and
        "max_{obs}" given as range to sample from.
        """
        for unit_type, column, idxs in self.sample_keys:
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

    def sample_state(self, net, *args, **kwargs):
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