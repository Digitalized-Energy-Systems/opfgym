""" Example how to implement a multi-stage OPF over multiple time steps.
Warning: Works only for simbench network because it requires timeseries data.
Warning: Creates only observation for the current time step, which means that
the agent has no prediction of the future and can only react to the current
state. In other words, long-term optimal actions are not necessarily possible
(and the Markov property is not fulfilled?).

TODO: Add this to the base class as a general method to handle multi-stage OPF?
TODO: Use all steps for observation? Essentially give the agent a prediction.
TODO: Add a storage system or something similar to this example to actually make the multi-stage aspect relevant.

"""


import pandapower as pp

from opfgym import opf_env
from opfgym.simbench.build_simbench_net import build_simbench_net


class MultiStageOpf(opf_env.OpfEnv):
    def __init__(self, simbench_network_name='1-LV-urban6--0-sw',
                 steps_per_episode=4,
                 train_sampler='simbench',
                 test_sampler='simbench',
                 validation_sampler='simbench',
                 *args, **kwargs):

        assert steps_per_episode > 1, "At least two steps required for a multi-stage OPF."
        assert 'simbench' in train_sampler and 'simbench' in test_sampler and 'simbench' in validation_sampler, "Only simbench networks are supported because time-series data required."

        net, profiles = self._define_opf(
            simbench_network_name, *args, **kwargs)

        # Observe all load power values
        obs_keys = [
            ('load', 'p_mw', net.load.index),
            ('load', 'q_mvar', net.load.index),
        ]

        # Control all generators in the system
        act_keys = [('sgen', 'p_mw', net.sgen.index)]

        super().__init__(net, act_keys, obs_keys,
                         profiles=profiles,
                         steps_per_episode=steps_per_episode,
                         optimal_power_flow_solver=False,
                         train_sampler=train_sampler,
                         test_sampler=test_sampler,
                         validation_sampler=validation_sampler,
                         *args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        net.sgen['controllable'] = True
        net.sgen['min_p_mw'] = net.sgen['min_min_p_mw']
        net.sgen['max_p_mw'] = net.sgen['max_max_p_mw']
        net.sgen['min_q_mvar'] = 0
        net.sgen['max_q_mvar'] = 0

        # Set everything else to uncontrollable
        for unit_type in ('load', 'gen', 'storage'):
            net[unit_type]['controllable'] = False

        # Objective: Minimize the active power lflow from external grid
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=1)

        return net, profiles

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

        # Increment the simbench step
        self._sampling(step=new_step)

        # Rerun the power flow calculation for the new state if required
        # TODO: This results in two power flow calculations for each step() call. Is it possible to avoid this?
        if self.pf_for_obs is True:
            self.run_power_flow()

        # Create new observation in the new state
        obs = self._get_obs(self.obs_keys, self.add_time_obs)

        return obs, reward, terminated, truncated, info


if __name__ == '__main__':
    env = MultiStageOpf()
    for _ in range(5):
        env.reset()
        env.step(env.action_space.sample())
