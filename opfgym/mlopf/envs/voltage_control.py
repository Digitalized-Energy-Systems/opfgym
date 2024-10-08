
import pandapower as pp

from opfgym import opf_env
from opfgym.simbench.build_simbench_net import build_simbench_net


class VoltageControl(opf_env.OpfEnv):
    """
    Voltage control environment to find the optimal reactive power setpoints to
    satisfy all constraints (especially voltage band) and to minimize losses
    within the system.

    Actuators: Reactive power of the bigger generators in the system.

    Sensors: Active+reactive power of all loads; active power of all generators
        and storages.

    Objective: minimize reactive power costs + minimize loss costs

    Constraints: Voltage band, line/trafo load, min/max reactive power,
        constrained reactive power flow over slack bus.

    """
    def __init__(self, simbench_network_name='1-MV-semiurb--1-sw',
                 load_scaling=1.3, gen_scaling=1.3, 
                 cos_phi=0.95, max_q_exchange=1.0, min_sgen_power=0.5, 
                 min_storage_power=0.5, market_based=False,
                 seed=None, *args, **kwargs):

        self.min_sgen_power = min_sgen_power
        self.min_storage_power = min_storage_power
        self.cos_phi = cos_phi
        self.market_based = market_based
        self.max_q_exchange = max_q_exchange
        self.net = self._define_opf(
            simbench_network_name, gen_scaling=gen_scaling,
            load_scaling=load_scaling, *args, **kwargs)

        # Define the RL problem
        # See all load power values, sgen/storage active power, and sgen prices...
        self.obs_keys = [
            ('sgen', 'p_mw', self.net.sgen.index),
            ('storage', 'p_mw', self.net.storage.index),
            ('load', 'p_mw', self.net.load.index),
            ('load', 'q_mvar', self.net.load.index),
        ]

        if market_based:
            # Consider reactive power prices in the objective function
            self.obs_keys.append(
                ('poly_cost', 'cq2_eur_per_mvar2', self.net.poly_cost.index)
            )

        # ... and control all units' reactive power values
        self.act_keys = [('sgen', 'q_mvar', self.net.sgen.index[self.net.sgen.controllable]),
                         ('storage', 'q_mvar', self.net.storage.index[self.net.storage.controllable])]

        super().__init__(seed=seed, *args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        net.load['controllable'] = False

        net.sgen['controllable'] = net.sgen.max_max_p_mw > self.min_sgen_power
        # Assumption: Generators can provide more reactive than active power
        net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / self.cos_phi
        net.sgen['max_max_q_mvar'] = net.sgen['max_s_mva']
        net.sgen['min_min_q_mvar'] = -net.sgen['max_s_mva']

        net.storage['controllable'] = net.storage.max_max_p_mw > self.min_storage_power
        # Assumption reactive power range = active power range
        net.storage['max_s_mva'] = net.storage['max_max_p_mw'].abs()
        net.storage['max_max_q_mvar'] = net.storage['max_s_mva']
        net.storage['min_min_q_mvar'] = -net.storage['max_s_mva']

        net.ext_grid['max_q_mvar'] = self.max_q_exchange
        net.ext_grid['min_q_mvar'] = -self.max_q_exchange

        # Add price params to the network (as poly cost so that the OPF works)
        # Add loss costs at slack so that objective = loss minimization
        # Comment: Costs are not in eur but eur/1000 instead
        self.loss_costs = 0.03
        for idx in net.sgen.index[net.sgen.controllable]:
            pp.create_poly_cost(net, idx, 'sgen',
                                cp1_eur_per_mw=self.loss_costs,
                                cq2_eur_per_mvar2=0)

        for idx in net.storage.index[net.storage.controllable]:
            pp.create_poly_cost(net, idx, 'storage',
                                cp1_eur_per_mw=-self.loss_costs,
                                cq2_eur_per_mvar2=0)

        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, 'ext_grid',
                                cp1_eur_per_mw=self.loss_costs,
                                cq2_eur_per_mvar2=0)

        assert len(net.gen) == 0  # TODO: Maybe add gens here, if necessary

        # Define range from which to sample reactive power prices on market
        self.max_price = 0.03
        net.poly_cost['min_cq2_eur_per_mvar2'] = 0
        net.poly_cost['max_cq2_eur_per_mvar2'] = self.max_price

        return net

    def _sampling(self, *args, **kwargs):
        super()._sampling(*args, **kwargs)

        # Sample reactive power prices uniformly from min/max range
        if self.market_based:
            for unit_type in ('sgen', 'ext_grid', 'storage'):
                self._sample_from_range(
                'poly_cost', 'cq2_eur_per_mvar2',
                self.net.poly_cost[self.net.poly_cost.et == unit_type].index)

        # Active power is not controllable (only relevant for OPF baseline)
        # Set active power boundaries to current active power values
        for unit_type in ('sgen', 'storage'):
            self.net[unit_type]['max_p_mw'] = self.net[unit_type].p_mw * self.net[unit_type].scaling + 1e-9
            self.net[unit_type]['min_p_mw'] = self.net[unit_type].p_mw * self.net[unit_type].scaling - 1e-9

        # Assumption: Generators offer all reactive power possible
        for unit_type in ('sgen', 'storage'):
            q_max = (self.net[unit_type].max_s_mva**2 - self.net[unit_type].max_p_mw**2)**0.5
            self.net[unit_type]['min_q_mvar'] = -q_max  # No scaling required this way!
            self.net[unit_type]['max_q_mvar'] = q_max
            # Make sure that without any action, zero Q is provided
            self.net[unit_type]['q_mvar'] = 0


if __name__ == '__main__':
    env = VoltageControl()
    print('VoltageControl environment created')
    print('Number of buses: ', len(env.net.bus))
    print('Observation space:', env.observation_space.shape)
    print('Action space:', env.action_space.shape, f'(Generators: {sum(env.net.sgen.controllable)}, Storage: {sum(env.net.storage.controllable)})')
