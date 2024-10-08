""" Simple example that demonstrates how to use non-Simbench power systems by
the example of the IEEE 30-bus system.
Warning: No time-series data is available for this network, which is the reason
that all benchmark environments use Simbench.
Warning: Still requires pandapower networks!"""


import pandapower as pp

from opfgym import opf_env


class NonSimbenchNet(opf_env.OpfEnv):
    def __init__(self, train_data='normal_around_mean',
                 test_data='normal_around_mean',
                 *args, **kwargs):

        assert 'simbench' not in train_data and 'simbench' not in test_data, "Only non-simbench networks are supported."

        self.net = self._define_opf()

        # Define the RL problem
        # Observe all load power values, sgen active power
        self.obs_keys = [
            ('load', 'p_mw', self.net.load.index),
            ('load', 'q_mvar', self.net.load.index),
        ]

        # ... and control some selected switches in the system
        self.act_keys = [('gen', 'p_mw', self.net.gen.index)]

        super().__init__(train_data=train_data, test_data=test_data,
                         *args, **kwargs)

    def _define_opf(self):
        # OPF problem already fully defined by pandapower
        net = pp.networks.case_ieee30()

        # Only active power actuators -> constrain reactive power to zero
        net.gen['min_q_mvar'] = 0
        net.gen['max_q_mvar'] = 0

        # Define data range for sampling of grid states
        range = 0.3
        # For uniform sampling, define range
        net.load['min_min_p_mw'] = net.load['p_mw'] * (1 - range)
        net.load['max_max_p_mw'] = net.load['p_mw'] * (1 + range)
        net.load['min_min_q_mvar'] = net.load['q_mvar'] * (1 - range)
        net.load['max_max_q_mvar'] = net.load['q_mvar'] * (1 + range)

        # For normal sampling, define mean and standard deviation
        net.load['mean_p_mw'] = net.load['p_mw']
        net.load['std_dev_p_mw'] = range * net.load['p_mw']
        net.load['mean_q_mvar'] = net.load['q_mvar']
        net.load['std_dev_q_mvar'] = range * net.load['q_mvar']

        # Also do this for the external grid (required for normalization later)
        net.ext_grid['mean_p_mw'] = net.load['mean_p_mw'].sum() - net.gen['p_mw'].sum()
        net.ext_grid['mean_q_mvar'] = net.load['mean_q_mvar'].sum() - (net.gen['max_q_mvar'] - net.gen['max_q_mvar']).sum()

        # TODO: Probably would be good to automatically do some of these steps in the base class?!
        # (They already happen automatically for Simbench networks!)

        return net


if __name__ == '__main__':
    env = NonSimbenchNet()
    for _ in range(5):
        env.reset()
        env.step(env.action_space.sample())
