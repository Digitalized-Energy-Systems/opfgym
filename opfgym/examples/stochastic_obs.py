

from opfgym.envs import VoltageControl
from opfgym.wrappers import StochasticObservation


def StochasticObs(noise_relative_range=0.1, *args, **kwargs):
    """ Create a simple stochastic OPF by adding noise to the observations
    without changing the underlying state."""
    # For some cases, noisy observations don't make sense -> use another implementation
    assert not ('add_mean_obs' in kwargs and kwargs['add_mean_obs'])
    assert not ('add_act_obs' in kwargs and kwargs['add_mean_obs'])
    assert not ('add_time_obs' in kwargs and kwargs['add_mean_obs'])

    env = VoltageControl(*args, **kwargs)
    wrapped_env = StochasticObservation(
        env,
        noise_relative_range=noise_relative_range,
        maintain_original_range=False)

    return wrapped_env
