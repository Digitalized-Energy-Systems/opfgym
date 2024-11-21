
import numpy as np
import pandapower as pp

from opfgym import opf_env
from opfgym.simbench.build_simbench_net import build_simbench_net
import opfgym.constraints as constraints


def get_s_mva_values(net: pp.pandapowerNet) -> np.ndarray:
    s_mva = (net.res_sgen.p_mw ** 2 + net.res_sgen.q_mvar ** 2) ** 0.5
    return s_mva

def get_s_mva_boundaries(net: pp.pandapowerNet) -> dict[str, np.ndarray]:
    """ Since this is a static constraint, it could also be defined in the 
    pandapower network. This is only for demonstration purposes."""
    return {'max': net.sgen.max_max_p_mw / 0.95}


class AddCustomConstraint(opf_env.OpfEnv):
    def __init__(self, simbench_network_name='1-LV-urban6--0-sw',
                 cos_phi=0.95, constraint_kwargs=None, *args, **kwargs):

        self.cos_phi = cos_phi
        net, profiles = self._define_opf(
            simbench_network_name, *args, **kwargs)

        obs_keys = [
            ('load', 'p_mw', net.load.index),
            ('load', 'q_mvar', net.load.index),
        ]

        act_keys = [('sgen', 'q_mvar', net.sgen.index)]

        # Extract default constraints (line loading, voltage band, etc.)
        constraint_kwargs = constraint_kwargs or {}
        constraints_list = constraints.create_default_constraints(
            net, constraint_kwargs)

        # Add custom constraint
        s_mva_constraint = constraints.Constraint(
            'sgen', 's_mva',
            get_values=get_s_mva_values,
            get_boundaries=get_s_mva_boundaries,
            **constraint_kwargs)
        constraints_list.append(s_mva_constraint)

        super().__init__(net, act_keys, obs_keys, profiles,
                         optimal_power_flow_solver=False,
                         constraints=constraints_list,
                         *args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        # Define reactive power of sgen as controllable
        net.sgen['controllable'] = True
        net.sgen['min_q_mvar'] = -0.3
        net.sgen['max_q_mvar'] = 0.3

        # Define new constraint in pandapower net
        net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / self.cos_phi

        # Set everything else to uncontrollable
        for unit_type in ('load', 'gen', 'storage'):
            net[unit_type]['controllable'] = False

        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=1)

        return net, profiles

    def _sampling(self, *args, **kwargs):
        super()._sampling(*args, **kwargs)

        # Active power is not controllable (only relevant for OPF baseline)
        # Set active power boundaries to current active power values
        self.net.sgen['max_p_mw'] = self.net.sgen.p_mw * self.net.sgen.scaling + 1e-9
        self.net.sgen['min_p_mw'] = self.net.sgen.p_mw * self.net.sgen.scaling - 1e-9



if __name__ == '__main__':
    env = AddCustomConstraint()
    for _ in range(5):
        env.reset()
        env.step(env.action_space.sample())
