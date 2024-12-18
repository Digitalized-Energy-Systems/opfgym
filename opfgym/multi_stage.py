
from opfgym import OpfEnv


class MultiStageOpfEnv(OpfEnv):
    """ Environment class for multi-stage OPF environments.

    Warning: Currently, the agent observes only the current step and does not
    receive any predictions about future states, which might result in
    suboptimal actions.

    Warning: Use simbench/time-series data sampling for training and testing
    so that incrementing the time step is possible.

    Args:
        Same as the base class

    """
    def __init__(self, *args, steps_per_episode: int = 4, **kwargs):
        assert steps_per_episode > 1, "At least two steps required for a multi-stage OPF."
        assert 'simbench' in kwargs.get('train_sampling', 'simbench')
        assert 'simbench' in kwargs.get('validation_sampling', 'simbench')
        assert 'simbench' in kwargs.get('test_sampling', 'simbench')
        super().__init__(*args, steps_per_episode=steps_per_episode, **kwargs)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if options:
            self.test = options.get('test', False)
        else:
            self.test = False


    def step(self, action):
        """ Extend step method to sample the next time step of the simbench data. """
        obs, reward, terminated, truncated, info = super().step(action)

        new_step = self.current_time_step + 1

        # Enforce train/test-split
        if self.test:
            # Do not accidentally test on train data!
            if new_step in self.train_steps:
                truncated = True
        else:
            # And do not accidentally train on test data!
            if new_step in self.validation_steps or new_step in self.test_steps:
                truncated = True

        # After n steps = end of episode
        if self.step_in_episode >= self.steps_per_episode:
            terminated = True

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Increment the time-series states
        self._sampling(step=new_step, test=self.test)

        # Rerun the power flow calculation for the new state if required
        if self.pf_for_obs is True:
            self.run_power_flow()

        # Create new observation in the new state
        obs = self._get_obs(self.obs_keys, self.add_time_obs)

        return obs, reward, terminated, truncated, info
