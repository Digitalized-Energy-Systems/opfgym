
import numpy as np
import pandapower as pp

from opfgym import opf_env
from opfgym.simbench.build_simbench_net import build_simbench_net


class ConstraintSatisfaction(opf_env.OpfEnv):
    def __init__(self, **kwargs):

        net, profiles = self._define_opf()

        # Define the RL problem
        # Observe all load power values, sgen active power
        obs_keys = [
            ('load', 'p_mw', net.load.index),
            ('load', 'q_mvar', net.load.index),
        ]

        # ... and control some selected switches in the system
        act_keys = [('sgen', 'p_mw', net.sgen.index)]

        super().__init__(net, act_keys, obs_keys, profiles=profiles, **kwargs)

    def _define_opf(self):
        net, profiles = build_simbench_net('1-LV-rural1--0-sw')

        net.sgen['controllable'] = True
        net.sgen['min_p_mw'] = 0
        net.sgen['max_p_mw'] = net.sgen['max_max_p_mw']
        net.sgen['min_q_mvar'] = 0
        net.sgen['max_q_mvar'] = 0

        # Set everything else to uncontrollable
        for unit_type in ('load', 'gen', 'storage'):
            net[unit_type]['controllable'] = False

        # Define some constraints
        net.ext_grid['max_p_mw'] = 1
        net.bus['max_vm_pu'] = 1.02
        net.bus['min_vm_pu'] = 0.98
        net.line['max_loading_percent'] = 60

        # Define no objective function

        return net, profiles
