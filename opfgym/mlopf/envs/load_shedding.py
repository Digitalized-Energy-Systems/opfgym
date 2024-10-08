
import numpy as np
import pandapower as pp

from opfgym import opf_env
from opfgym.simbench.build_simbench_net import build_simbench_net


# Problem: Sadly pandapower cannot deal with coupled active/reactive power,
# which would be necessary for this environment.
# Current solution: Only active power shedding...

class LoadShedding(opf_env.OpfEnv):
    """
    Load shedding environment to find the load active power setpoints that
    satisfy all system constraints at minimal costs.

    Actuators: Active power of all bigger loads and storages.

    Sensors: active+reactive power of all loads; active power of all generators;
        prices for load shedding and storage usage.

    Objective: Minimize costs (naive solution: no load shedding).

    Constraints: Voltage band, line/trafo load, min/max active power limits
        (automatically considered), active power exchange with external grid.

    """

    def __init__(self, simbench_network_name='1-MV-comm--2-sw',
                 gen_scaling=1.6, load_scaling=2.2, min_load_power=0.6,
                 min_storage_power=1.0, max_p_exchange=8.0, *args, **kwargs):

        self.min_load_power = min_load_power
        self.min_storage_power = min_storage_power
        self.max_p_exchange = max_p_exchange
        self.net = self._define_opf(
            simbench_network_name, gen_scaling=gen_scaling,
            load_scaling=load_scaling, *args, **kwargs)

        # Define the RL problem
        # See all load power values, sgen max active power...
        self.obs_keys = [('sgen', 'p_mw', self.net.sgen.index),
                         ('load', 'p_mw', self.net.load.index),
                         ('load', 'q_mvar', self.net.load.index),
                         ('storage', 'p_mw', self.net.storage.index[~self.net.storage.controllable]),
                         ('poly_cost', 'cp1_eur_per_mw', self.net.poly_cost.index)]

        # Control active power of loads and storages
        self.act_keys = [('load', 'p_mw', self.net.load.index[self.net.load.controllable]),
                         ('storage', 'p_mw', self.net.storage.index[self.net.storage.controllable])]

        super().__init__(*args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        net.load['controllable'] = net.load.max_max_p_mw > self.min_load_power
        # Assumption: Every load can be reduced to zero power
        net.load['min_min_p_mw'] = 0
        net.load['min_p_mw'] = 0

        # Biggest Storage system controllable
        max_storage_power = np.maximum(
            net.storage['min_min_p_mw'].abs(), net.storage['max_max_p_mw'].abs())
        net.storage['min_p_mw'] = -max_storage_power
        net.storage['max_p_mw'] = max_storage_power
        net.storage['min_min_p_mw'] = -max_storage_power
        net.storage['max_max_p_mw'] = max_storage_power
        net.storage['controllable'] = net.storage.max_max_p_mw > self.min_storage_power

        net.sgen['controllable'] = False

        # Constraint: We can not have too much active power from ext grid
        net.ext_grid['max_p_mw'] = self.max_p_exchange
        net.ext_grid['min_p_mw'] = -np.inf

        for unit_type in ('load', 'storage'):
            for idx in net[unit_type].index[net[unit_type].controllable]:
                pp.create_poly_cost(net, idx, unit_type, cp1_eur_per_mw=0)

        # Define range from which to sample load shedding prices
        # Negative costs, because higher=better (less load shedding)
        max_load_shedding_price = 10
        net.poly_cost['min_cp1_eur_per_mw'] = -max_load_shedding_price
        net.poly_cost['max_cp1_eur_per_mw'] = 0
        # Assumption: using storage is far cheaper on average
        # Assumption: Perfect storage efficiency
        max_storage_price = 2
        net.poly_cost['min_cp1_eur_per_mw'][net.poly_cost.et == 'storage'] = 0
        net.poly_cost['max_cp1_eur_per_mw'][net.poly_cost.et == 'storage'] = max_storage_price

        net.ext_grid['vm_pu'] = 1.0

        return net

    def _sampling(self, *args, **kwargs):
        super()._sampling(*args, **kwargs)

        # Sample prices for loads and storages
        # The idea is that not always the same loads should be shedded
        self._sample_from_range(
            'poly_cost', 'cp1_eur_per_mw', self.net.poly_cost.index)

        # Current load power = maximum power (only reduction possible)
        self.net.load['max_p_mw'] = self.net.load['p_mw'] * self.net.load.scaling + 1e-9

        # Make sure reactive power is not controllable
        for unit_type in ('load', 'storage'):
            self.net[unit_type]['max_q_mvar'] = self.net[unit_type].q_mvar * self.net[unit_type].scaling + 1e-9
            self.net[unit_type]['min_q_mvar'] = self.net[unit_type].q_mvar * self.net[unit_type].scaling - 1e-9


if __name__ == '__main__':
    env = LoadShedding()
    print('Load shedding environment created')
    print('Number of buses: ', len(env.net.bus))
    print('Observation space:', env.observation_space.shape)
    print('Action space:', env.action_space.shape, f'(Loads: {sum(env.net.load.controllable)}, Storage: {sum(env.net.storage.controllable)})')
