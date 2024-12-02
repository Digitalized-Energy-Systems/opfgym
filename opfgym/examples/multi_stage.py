""" Example how to implement a multi-stage OPF over multiple time steps.
Warning: Works only for simbench network because it requires timeseries data.
Warning: Creates only observation for the current time step, which means that
the agent has no prediction of the future and can only react to the current
state. In other words, long-term optimal actions are not necessarily possible
(and the Markov property is not fulfilled).

TODO: Use all steps for observation? Essentially give the agent a prediction.
TODO: Add a storage system or something similar to actually make the multi-stage aspect relevant.

"""


import pandapower as pp

from opfgym import MultiStageOpfEnv
from opfgym.simbench.build_simbench_net import build_simbench_net


class MultiStageOpf(MultiStageOpfEnv):
    def __init__(self, simbench_network_name='1-LV-urban6--0-sw',
                 steps_per_episode=4, train_sampling='simbench',
                 test_sampling='simbench', validation_sampling='simbench',
                 *args, **kwargs):

        assert steps_per_episode > 1, "At least two steps required for a multi-stage OPF."
        assert 'simbench' in train_sampling, "Only simbench networks are supported because time-series data required."
        assert 'simbench' in test_sampling, "Only simbench networks are supported because time-series data required."
        assert 'simbench' in validation_sampling, "Only simbench networks are supported because time-series data required."

        net, profiles = self._define_opf(
            simbench_network_name, *args, **kwargs)

        # Observe all load power values
        obs_keys = [
            ('load', 'p_mw', net.load.index),
            ('load', 'q_mvar', net.load.index),
        ]

        # Control all generators in the system
        act_keys = [('sgen', 'p_mw', net.sgen.index)]

        super().__init__(net, act_keys, obs_keys, profiles=profiles,
                         steps_per_episode=steps_per_episode,
                         optimal_power_flow_solver=False,
                         train_sampling=train_sampling,
                         test_sampling=test_sampling,
                         validation_sampling=validation_sampling,
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


if __name__ == '__main__':
    env = MultiStageOpf()
    for _ in range(5):
        env.reset()
        env.step(env.action_space.sample())
