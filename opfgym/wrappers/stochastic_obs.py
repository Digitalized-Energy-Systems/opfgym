
from gymnasium import spaces
from gymnasium import ObservationWrapper
import numpy as np


# TODO: Probably good idea to add noise only to part of the observations
# (e.g., only to power setpoints but not to price information)

class StochasticObservation(ObservationWrapper):
    def __init__(self,
                 env,
                 noise_relative_range: float=0.1,
                 maintain_original_range: bool=True):
        """ Adds noise to the observations to create stochastic observations
        without changing the underlying state. Results in a simple stochastic 
        OPF.

        Args:
            noise_relative_range (float): The noise range relative to the
                observation space range (uniform distribution).
            maintain_original_range (bool): If True, the observations are
                clipped to the original observation space range. If False, the
                observation space is expanded to include the noise range. Set
                this to True if a wider obs range does not make sense, e.g.,
                negative loads or generation above the max capacity.
        """
        super().__init__(env)
        self.maintain_original_range = maintain_original_range

        obs_range = env.observation_space.high - env.observation_space.low
        self.abs_noise_range = noise_relative_range * obs_range

        if not maintain_original_range:
            low_observation = env.observation_space.low - self.abs_noise_range
            high_observation = env.observation_space.high + self.abs_noise_range
            self.observation_space = spaces.Box(low=low_observation,
                                                high=high_observation)

    def observation(self, observation):
        noise = self.np_random.uniform(-self.abs_noise_range,
                                       self.abs_noise_range,
                                       size=observation.shape)

        observation += noise

        if self.maintain_original_range:
            observation = np.clip(observation,
                                  self.observation_space.low,
                                  self.observation_space.high)

        return observation
