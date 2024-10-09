""" Simple example that shows how to define a partially observable RL-OPF
environment. Especially useful for real-world applications where not all 
measurement data will be available. """


import numpy as np
import pandapower as pp

from opfgym import opf_env
from opfgym.simbench.build_simbench_net import build_simbench_net


class PartiallyObservable(opf_env.OpfEnv):
    def __init__(self, simbench_network_name='1-LV-rural1--0-sw',
                 observable_loads=np.arange(10),  # First 20 loads are observable
                 *args, **kwargs):

        self.net = self._define_opf(
            simbench_network_name, *args, **kwargs)

        if isinstance(observable_loads, str) and observable_loads == 'all':
            observable_loads = self.net.load.index

        # Define the RL problem
        # Observe all load power values, sgen active power
        self.obs_keys = [
            ('load', 'p_mw', observable_loads),  # Only observe selected loads
            ('load', 'q_mvar', observable_loads),
        ]

        # ... and control some selected switches in the system
        self.act_keys = [('sgen', 'p_mw', self.net.sgen.index)]

        super().__init__(*args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        net.sgen['controllable'] = True
        net.sgen['min_p_mw'] = 0
        net.sgen['max_p_mw'] = net.sgen['max_max_p_mw']
        net.sgen['min_q_mvar'] = 0
        net.sgen['max_q_mvar'] = 0

        # Set everything else to uncontrollable
        for unit_type in ('load', 'gen', 'storage'):
            net[unit_type]['controllable'] = False

        # Objective: Minimize the active power losses
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=1)

        return net


if __name__ == '__main__':
    env = PartiallyObservable()
    for _ in range(5):
        env.reset()
        env.step(env.action_space.sample())
