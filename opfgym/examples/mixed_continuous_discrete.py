""" Simple example that shows how using continuous and discrete actuatory in the
same environment is easily possible. This is a nice demonstration how something
that is very difficult to model with conventional solvers can be easilty modeled
and solved as an RL environment.
Note that from an RL perspective, all actions are modeled as continuous actions,
which requires algorithms like PPO, SAC, etc.
This example also shows how to easily overwrite the objective function with an
objective that is not pandapower-compatible. """


import pandapower as pp
import numpy as np

from opfgym import opf_env
from opfgym.simbench.build_simbench_net import build_simbench_net


class MixedContinuousDiscrete(opf_env.OpfEnv):
    def __init__(self, simbench_network_name='1-LV-urban6--0-sw',
                 cos_phi=0.95, *args, **kwargs):

        self.cos_phi = cos_phi
        self.net = self._define_opf(
            simbench_network_name, *args, **kwargs)

        # Define the RL problem
        # Observe all load power values, sgen active power
        self.obs_keys = [
            ('ext_grid', 'vm_pu', self.net.ext_grid.index),  # Slack voltage
            ('sgen', 'p_mw', self.net.sgen.index),
            ('load', 'p_mw', self.net.load.index),
            ('load', 'q_mvar', self.net.load.index),
        ]

        # ... and control trafos and reactive power for voltage control
        self.act_keys = [('sgen', 'q_mvar', self.net.sgen.index),
                         ('trafo', 'tap_pos', self.net.trafo.index)]

        super().__init__(*args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        # Define the transformer tap positions as controllable
        net.trafo['controllable'] = True
        net.trafo['min_tap_pos'] = -2
        net.trafo['max_tap_pos'] = 2

        # Define reactive power of sgen as controllable
        net.sgen['controllable'] = True
        net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / self.cos_phi
        net.sgen['max_max_q_mvar'] = (net.sgen['max_s_mva']**2 - net.sgen['max_max_p_mw']**2)**0.5
        net.sgen['min_min_q_mvar'] = -net.sgen['max_max_q_mvar']
        net.sgen['max_q_mvar'] = net.sgen['max_max_q_mvar']
        net.sgen['min_q_mvar'] = -net.sgen['max_max_q_mvar']

        # Set everything else to uncontrollable
        for unit_type in ('load', 'gen', 'storage'):
            net[unit_type]['controllable'] = False

        # Define voltage range of the slack bus (ext_grid)
        net.ext_grid['min_vm_pu'] = 0.95
        net.ext_grid['max_vm_pu'] = 1.05

        return net

    def _sampling(self, *args, **kwargs):
        super()._sampling(*args, **kwargs)

        # Sample slack voltage randomly to make the problem more difficult
        # so that trafo tap changing is required for voltage control
        self._sample_from_range('ext_grid', 'vm_pu', self.net.ext_grid.index)

        # Active power is not controllable (only relevant for OPF baseline)
        # Set active power boundaries to current active power values
        for unit_type in ('sgen',):
            self.net[unit_type]['max_p_mw'] = self.net[unit_type].p_mw * self.net[unit_type].scaling + 1e-9
            self.net[unit_type]['min_p_mw'] = self.net[unit_type].p_mw * self.net[unit_type].scaling - 1e-9

    def calculate_base_objective(self, net) -> np.ndarray:
        """ Use quadratic voltage deviation from 1.0 pu as objective."""
        return -(net.res_bus.vm_pu - 1)**2

    def get_optimal_objective(self):
        # Overwrite because not solvable with pandapower OPF solver
        return 0

    def run_optimal_power_flow(self, **kwargs):
        # Overwrite because not solvable with pandapower OPF solver
        return False


if __name__ == '__main__':
    env = MixedContinuousDiscrete()
    for _ in range(5):
        env.reset()
        env.step(env.action_space.sample())
