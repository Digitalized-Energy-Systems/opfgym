import math

import pandapower as pp

from mlopf import opf_env
from mlopf.build_simbench_net import build_simbench_net


class MaxRenewable(opf_env.OpfEnv):
    """
    The goal is to learn to set active and reactive power of all generators in 
    the system to maximize active power feed-in to the external grid.
    Since this environment has an obvious solution to the optimization problem 
    (set all gens to 100% power), it is well suited to investigate constraint 
    satisfaction.

    Actuators: Active/reactive power of all generators

    Sensors: active+reactive power of all loads; max active power of all gens

    Objective: maximize active power feed-in to external grid

    Constraints: Voltage band, line/trafo load, min/max reactive power,
        constrained reactive power flow over slack bus
    """

    def __init__(self, simbench_network_name='1-LV-rural3--2-sw',
                 gen_scaling=3.0, load_scaling=1.0, cos_phi=0.95,
                 storage_efficiency=0.95,
                 seed=None,
                 *args, **kwargs):

        self.cos_phi = cos_phi
        self.storage_efficiency = storage_efficiency
        self.net = self._define_opf(
            simbench_network_name, gen_scaling=gen_scaling,
            load_scaling=load_scaling, *args, **kwargs)

        # Define the RL problem
        # See all load power values, sgen max active power...
        self.obs_keys = [('sgen', 'p_mw', self.net.sgen.index),
                         ('load', 'p_mw', self.net.load.index),
                         ('load', 'q_mvar', self.net.load.index)]

        # ... and control all sgens' and storages' active power values
        self.act_keys = [('sgen', 'p_mw', self.net.sgen.index),
                         ('storage', 'p_mw', self.net.storage.index)]

        if 'ext_grid_pen_kwargs' not in kwargs:
            kwargs['ext_grid_pen_kwargs'] = {'linear_penalty': 25}
        if 'volt_pen_kwargs' not in kwargs:
            kwargs['volt_pen_kwargs'] = {'linear_penalty': 5}
        super().__init__(seed=seed, *args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        net.load['controllable'] = False

        net.storage['controllable'] = True 
        net.storage['q_mvar'] = 0
        net.storage['max_q_mvar'] = 0
        net.storage['min_q_mvar'] = 0
        # Assume that storage systems are completely usable 
        # (do not consider state of charge for example)
        net.storage['max_p_mw'] = net.storage['max_max_p_mw']
        net.storage['min_p_mw'] = net.storage['min_min_p_mw']
        
        net.sgen['controllable'] = True
        net.sgen['min_p_mw'] = 0  # max will be set later in sampling
        net.sgen['q_mvar'] = 0
        net.sgen['max_q_mvar'] = 0
        net.sgen['min_q_mvar'] = 0

        # Assumption: Mandatory reactive power provision of cos_phi
        self.q_factor = -math.tan(math.acos(self.cos_phi))

        # OPF objective: Maximize active power feed-in to external grid
        # TODO: Maybe allow for gens here, if necessary
        assert len(net.gen) == 0, 'gen not supported in this environment!'
        active_power_costs = 30
        for idx in net['sgen'].index:
            pp.create_poly_cost(net, idx, 'sgen',
                                cp1_eur_per_mw=-active_power_costs)
            
        # Assumption: Storage power is more expensive than sgen power
        storage_costs = active_power_costs / (self.storage_efficiency**2)
        for idx in net['storage'].index:
            pp.create_poly_cost(net, idx, 'storage',
                                cp1_eur_per_mw=storage_costs)

        return net

    def _sampling(self, *args, **kwargs):
        super()._sampling(*args, **kwargs)
        # TODO: Maybe add storage max power here, e.g., to consider the current state of charge?!

        # Set constraints of current time step (also required for OPF)
        self.net.sgen['max_p_mw'] = self.net.sgen.p_mw * self.net.sgen.scaling + 1e-6

        self.net.sgen['q_mvar'] = self.net.sgen.p_mw * self.q_factor
        self.net['max_q_mvar'] = self.net.sgen.q_mvar * self.net.sgen.scaling + 1e-9
        self.net['min_q_mvar'] = self.net.sgen.q_mvar * self.net.sgen.scaling - 1e-9


if __name__ == '__main__':
    env = MaxRenewable()
    print('Max renewable environment created')
    print('Observation space:', env.observation_space.shape)
    print('Action space:', env.action_space.shape)
