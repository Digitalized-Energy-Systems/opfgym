
import copy

import numpy as np
import pandapower as pp

from opfgym import opf_env
from opfgym.simbench.build_simbench_net import build_simbench_net


class SecurityConstrained(opf_env.OpfEnv):
    def __init__(self, simbench_network_name='1-HV-urban--0-sw',
                 n_minus_one_lines=(1, 3, 7),
                 *args, **kwargs):
        self.n_minus_one_lines = np.array(n_minus_one_lines)

        self.net = self._define_opf(
            simbench_network_name, *args, **kwargs)

        # Define the RL problem
        # Observe all load power values, sgen active power
        self.obs_keys = [
            ('load', 'p_mw', self.net.load.index),
            ('load', 'q_mvar', self.net.load.index),
        ]

        # ... and control some selected switches in the system
        self.act_keys = [('sgen', 'p_mw', self.net.sgen.index)]

        super().__init__(*args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
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

        return net

    def calculate_violations(self, original_net=None):
        """ Implement the security constrained power flow by removing the n-1 lines and checking for violations. """
        original_net = original_net if original_net else self.net
        valids, viol, penalties = super().calculate_violations(original_net)

        # Remove singular lines and check for violations afterward
        for line_idx in self.n_minus_one_lines:
            net = copy.deepcopy(original_net)
            # Remove the line
            pp.drop_lines(net, [line_idx])
            # Run the power flow (TODO: better use the build-in method -> change API)
            pp.runpp(net)
            # Check for violations
            new_valids, new_viol, new_penalties = super().calculate_violations(net)
            # Update the violations
            valids = np.logical_and(valids, new_valids)
            viol += new_viol
            penalties += new_penalties

        return valids, viol, penalties

    def get_optimal_objective(self):
        # Overwrite because not solvable with pandapower OPF solver
        return 0

    def run_optimal_power_flow(self, **kwargs):
        # Overwrite because not solvable with pandapower OPF solver
        return False


if __name__ == '__main__':
    env = SecurityConstrained()
    for _ in range(5):
        env.reset()
        env.step(env.action_space.sample())
