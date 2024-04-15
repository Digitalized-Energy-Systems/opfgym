""" 
Load shedding environment: The agent learns to perform cost minimal load 
shedding in case of a grid overload.

TODO: How to deal with reactive power actuators?
TODO: Maybe change this to demand response env (load can be increased as well)

"""

import numpy as np
import pandapower as pp

from mlopf import opf_env
from mlopf.build_simbench_net import build_simbench_net


# TODO: Problem: Sadly pandapower cannot deal with coupled active/reactive power,
# which would be necassary for this environment. Current solution: Only active power shedding...

class LoadShedding(opf_env.OpfEnv):
    """
    Description: TODO

    Actuators: Active/reactive (coupled) power of all loads, active and reactive 
        power of all storages

    Sensors: active+reactive power of all loads; active power of all gens

    Objective: Minimize costs

    Constraints: Voltage band, line/trafo load, ext_grid max active power

    """

    def __init__(self, simbench_network_name='1-LV-semiurb4--2-sw',
                 gen_scaling=1.0, load_scaling=3.0,
                 max_p_exchange=0.1, *args, **kwargs):

        self.max_p_exchange = max_p_exchange
        self.net = self._define_opf(
            simbench_network_name, gen_scaling=gen_scaling,
            load_scaling=load_scaling, *args, **kwargs)

        # Define the RL problem
        # See all load power values, sgen max active power...
        self.obs_keys = [('sgen', 'p_mw', self.net.sgen.index),
                         ('load', 'p_mw', self.net.load.index),
                         ('load', 'q_mvar', self.net.load.index),
                         ('poly_cost', 'cp1_eur_per_mw', self.net.poly_cost.index)]

        # Control active power of loads and storages
        self.act_keys = [('load', 'p_mw', self.net.load.index),
                         ('storage', 'p_mw', self.net.storage.index)]

        # Define default penalties
        if 'ext_grid_pen_kwargs' not in kwargs:
            kwargs['ext_grid_pen_kwargs'] = {'linear_penalty': 25}

        super().__init__(*args, **kwargs)


    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        net.load['controllable'] = True
        # Assumption: Every load can be reduced to zero power
        net.load['min_min_p_mw'] = 0
        net.load['min_p_mw'] = 0

        # Storages fully controllable
        net.storage['controllable'] = True
        max_storage_power = np.maximum(
            net.storage['min_min_p_mw'].abs(), net.storage['max_max_p_mw'].abs())
        net.storage['min_p_mw'] = -max_storage_power
        net.storage['max_p_mw'] = max_storage_power

        net.sgen['controllable'] = False

        # Constraint: We can not have too much active power from ext grid
        net.ext_grid['max_p_mw'] = self.max_p_exchange
        net.ext_grid['min_p_mw'] = -np.inf

        for unit_type in ('load', 'storage'):
            for idx in net[unit_type].index:
                pp.create_poly_cost(net, idx, unit_type, cp1_eur_per_mw=0)

        # Define range from which to sample load shedding prices
        # Negative costs, because higher=better (less load shedding)
        max_load_shedding_price = 10
        net.poly_cost['min_cp1_eur_per_mw'] = -max_load_shedding_price
        net.poly_cost['max_cp1_eur_per_mw'] = 0
        # Assumption: using storage is far cheaper on average
        max_storage_price = 1
        net.poly_cost['min_cp1_eur_per_mw'][net.poly_cost.et == 'storage'] = 0
        net.poly_cost['max_cp1_eur_per_mw'][net.poly_cost.et == 'storage'] = max_storage_price

        return net

    def _sampling(self, *args, **kwargs):
        super()._sampling(*args, **kwargs)

        # Sample prices for loads and storages
        for unit_type in ('load', 'storage'):
            self._sample_from_range(
            'poly_cost', 'cp1_eur_per_mw',
            self.net.poly_cost[self.net.poly_cost.et == unit_type].index)

        # Current load power = maximum power (only reduction possible)
        self.net.load['max_p_mw'] = self.net.load['p_mw'] * self.net.load.scaling

        # Make sure reactive power is not controllable
        for unit_type in ('load', 'storage'):
            self.net[unit_type]['max_q_mvar'] = self.net[unit_type].q_mvar * self.net[unit_type].scaling + 1e-9
            self.net[unit_type]['min_q_mvar'] = self.net[unit_type].q_mvar * self.net[unit_type].scaling - 1e-9


if __name__ == '__main__':
    env = LoadShedding()
    print('Load shedding environment created')
    print('Observation space:', env.observation_space.shape)
    print('Action space:', env.action_space.shape)
    