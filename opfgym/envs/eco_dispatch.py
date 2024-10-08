
import pandapower as pp

from opfgym import opf_env
from opfgym.simbench.build_simbench_net import build_simbench_net


class EcoDispatch(opf_env.OpfEnv):
    """
    Economic Dispatch/Active power market environment: The grid operator
    procures active power from generators to satisfy all constraints at minimal
    costs.

    Actuators: Active power of the larger generators in the system.

    Sensors: active+reactive power of all loads/storages;
        active power prices of the controllable generators;
        active power setpoints of the non-controllable generators.

    Objective: minimize active power costs.

    Constraints: Voltage band, line/trafo load, min/max active power limits
        (automatically considered), active power exchange with external grid.

    """

    def __init__(self, simbench_network_name='1-HV-urban--0-sw', 
                 gen_scaling=1.0, load_scaling=1.5, max_price_eur_gwh=0.5,
                 min_power=0, seed=None, *args, **kwargs):

        # Define range from which to sample active power prices on market
        self.max_price_eur_gwh = max_price_eur_gwh
        # compare: https://en.wikipedia.org/wiki/Cost_of_electricity_by_source

        # Minimal power to be considered as an actuator for the eco dispatch
        self.min_power = min_power

        self.net = self._define_opf(
            simbench_network_name, gen_scaling=gen_scaling,
            load_scaling=load_scaling, *args, **kwargs)

        # Define the RL problem
        # See all load power values, non-controlled generators, and generator prices...
        self.obs_keys = [('load', 'p_mw', self.net.load.index),
                         ('load', 'q_mvar', self.net.load.index),
                         ('poly_cost', 'cp1_eur_per_mw', self.net.poly_cost.index),
                         ('sgen', 'p_mw', self.net.sgen.index[~self.net.sgen.controllable]),
                         ('storage', 'p_mw', self.net.storage.index),
                         ('storage', 'q_mvar', self.net.storage.index)]

        # ... and control all generators' active power values
        self.act_keys = [('sgen', 'p_mw', self.net.sgen.index[self.net.sgen.controllable]),
                         ('gen', 'p_mw', self.net.gen.index[self.net.gen.controllable])]

        super().__init__(seed=seed, *args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)
        # TODO: Set voltage setpoints a bit higher than 1.0 to consider voltage drop?
        net.ext_grid['vm_pu'] = 1.0
        net.gen['vm_pu'] = 1.0

        net.load['controllable'] = False

        # Prevent "selling" of active power to upper system
        net.ext_grid['min_p_mw'] = 0
        # Prevent infinite power consumption. Set to biggest generator.
        net.ext_grid['max_p_mw'] = net.sgen.max_max_p_mw.max()

        # Generator constraints required for OPF
        net.sgen['min_p_mw'] = 0
        net.sgen['max_p_mw'] = net.sgen['max_max_p_mw']
        net.gen['min_p_mw'] = 0
        net.gen['max_p_mw'] = net.gen['max_max_p_mw']

        # Define which generators are controllable
        net.sgen['controllable'] = net.sgen.max_max_p_mw > self.min_power
        net.sgen['min_min_p_mw'] = 0
        net.gen['controllable'] = True

        # Completely neglect reactive power
        for unit_type in ('gen', 'sgen'):
            net[unit_type]['max_q_mvar'] = 0.0
            net[unit_type]['min_q_mvar'] = 0.0

        # Add price params to the network (as poly cost so that the OPF works)
        # Note that the external grids are seen as normal power plants
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=0)
        for idx in net.sgen.index[net.sgen.controllable]:
            pp.create_poly_cost(net, idx, 'sgen', cp1_eur_per_mw=0)
        for idx in net.gen.index[net.gen.controllable]:
            pp.create_poly_cost(net, idx, 'gen', cp1_eur_per_mw=0)

        net.poly_cost['min_cp1_eur_per_mw'] = 0
        net.poly_cost['max_cp1_eur_per_mw'] = self.max_price_eur_gwh

        return net

    def _sampling(self, *args, **kwargs):
        super()._sampling(*args, **kwargs)

        # Sample prices uniformly from min/max range for gens/sgens/ext_grids
        self._sample_from_range(
            'poly_cost', 'cp1_eur_per_mw', self.net.poly_cost.index)


if __name__ == '__main__':
    env = EcoDispatch()
    print('EcoDispatch environment created')
    print('Number of buses: ', len(env.net.bus))
    print('Observation space:', env.observation_space.shape)
    print('Action space:', env.action_space.shape)
