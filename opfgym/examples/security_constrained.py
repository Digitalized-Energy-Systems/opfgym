
import numpy as np
import pandapower as pp

from opfgym.simbench.build_simbench_net import build_simbench_net
from opfgym import SecurityConstrainedOpfEnv

# Inherit from the SecurityConstrainedOpfEnv class to automatically add the
# n-1 security constraint for all considered constraints
class SecurityConstrained(SecurityConstrainedOpfEnv):
    def __init__(self, simbench_network_name='1-HV-urban--0-sw',
                 *args, **kwargs):
        n_minus_one_keys = (('line', 'in_service', np.array([1, 3, 7])),)

        net, profiles = self._define_opf(
            simbench_network_name, *args, **kwargs)

        # Define the RL problem
        # Observe all load power values, sgen active power
        obs_keys = [
            ('load', 'p_mw', net.load.index),
            ('load', 'q_mvar', net.load.index),
        ]

        # ... and control some selected switches in the system
        act_keys = [('sgen', 'p_mw', net.sgen.index)]

        super().__init__(net, act_keys, obs_keys,
                         n_minus_one_keys=n_minus_one_keys,
                         profiles=profiles,
                         optimal_power_flow_solver=False, *args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        net.sgen['controllable'] = True
        net.sgen['max_p_mw'] = net.sgen['max_max_p_mw']
        net.sgen['min_p_mw'] = net.sgen['min_min_p_mw']
        net.sgen['max_q_mvar'] = 0
        net.sgen['min_q_mvar'] = 0

        # Set everything else to uncontrollable
        for unit_type in ('load', 'gen', 'storage'):
            net[unit_type]['controllable'] = False

        # Objective: Minimize the active power losses
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=0.01)

        return net, profiles


if __name__ == '__main__':
    env = SecurityConstrained()
    for _ in range(5):
        env.reset()
        env.step(env.action_space.sample())
