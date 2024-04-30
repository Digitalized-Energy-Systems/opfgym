
import gymnasium as gym
import numpy as np
import pandapower as pp

from mlopf import opf_env
from mlopf.build_simbench_net import build_simbench_net


class EcoDispatch(opf_env.OpfEnv):
    """
    Economic Dispatch/Active power market environment: The grid operator
    procures active power from generators to minimize losses within its system.

    Actuators: Active power of all gens

    Sensors: active+reactive power of all loads; (TODO: active power of all other gens?);
        active power prices of all gens

    Objective: minimize active power costs

    Constraints: Voltage band, line/trafo load, min/max active power limits
        (automatically), active power exchange with external grid

    """

    def __init__(self, simbench_network_name='1-HV-urban--0-sw', min_power=0,
                 n_agents=None, gen_scaling=1.0, load_scaling=1.5, max_price=600,
                 seed=None, *args, **kwargs):
        

        # Economic dispatch normally done in EHV (too big! use HV instead!)
        # EHV option: '1-EHV-mixed--0-sw' (340 generators!!!)
        # HV options: '1-HV-urban--0-sw' and '1-HV-mixed--0-sw'

        # Not every power plant is big enough to participate in the market
        # Assumption: Use power from time-series for all other plants (see sampling())
        # Set min_power=0 to consider all power plants as market participants
        # Alternatively use n_agents to use the n_agents biggest power plants

        # Define range from which to sample active power prices on market
        self.max_price = max_price
        # compare: https://en.wikipedia.org/wiki/Cost_of_electricity_by_source

        self.net = self._define_opf(
            simbench_network_name, min_power, n_agents, gen_scaling=gen_scaling,
            load_scaling=load_scaling, *args, **kwargs)

        # Define the RL problem
        # See all load power values, non-controlled generators, and generator prices...
        # non_sgen_idxs = self.net.sgen.index.drop(self.sgen_idxs)
        # non_gen_idxs = self.net.gen.index.drop(self.gen_idxs)
        # bid_idxs = np.array(
        #     range(len(self.sgen_idxs) + len(self.gen_idxs) + len(self.net.ext_grid.index)))
        self.obs_keys = [('load', 'p_mw', self.net.load.index),
                         ('load', 'q_mvar', self.net.load.index),
                         # ('res_sgen', 'p_mw', non_sgen_idxs),
                         # ('res_gen', 'p_mw', non_gen_idxs),
                         ('poly_cost', 'cp1_eur_per_mw', self.net.poly_cost.index)]

        # ... and control all generators' active power values
        self.act_keys = [('sgen', 'p_mw', self.net.sgen.index),  # self.sgen_idxs),
                         ('gen', 'p_mw', self.net.gen.index)]  # self.gen_idxs)]
        # TODO: Define constraints explicitly?! (active power min/max not default!)

        # Set default values
        if 'line_pen_kwargs' not in kwargs:
            kwargs['line_pen_kwargs'] = {'linear_penalty': 3000}
        if 'trafo_pen_kwargs' not in kwargs:
            kwargs['trafo_pen_kwargs'] = {'linear_penalty': 3000}
        if 'ext_grid_pen_kwargs' not in kwargs:
            kwargs['ext_grid_pen_kwargs'] = {'linear_penalty': 500000}

        super().__init__(seed=seed, *args, **kwargs)

        if self.vector_reward is True:
            # 5 penalties and `n_participants` objective functions
            n_objs = 5 + len(self.net.sgen) + \
                len(self.net.ext_grid) + len(self.net.gen)
            self.reward_space = gym.spaces.Box(
                low=-np.ones(n_objs) * np.inf,
                high=np.ones(n_objs) * np.inf,
                seed=seed)

    def _define_opf(self, simbench_network_name, min_power, n_agents, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)
        # Set voltage setpoints a bit higher than 1.0 to consider voltage drop?
        net.ext_grid['vm_pu'] = 1.0
        net.gen['vm_pu'] = 1.0

        net.load['controllable'] = False

        # Generator constraints required for OPF!
        net.sgen['min_p_mw'] = 0
        net.sgen['max_p_mw'] = net.sgen['max_max_p_mw']
        net.gen['min_p_mw'] = 0
        net.gen['max_p_mw'] = net.gen['max_max_p_mw']

        # Prevent "selling" of active power to upper system
        net.ext_grid['min_p_mw'] = 0

        # TODO: Also for gen
        #     axis=0) * net['sgen']['scaling']
        # net.sgen['min_max_p_mw'] = 0
        net.sgen['controllable'] = True
        net.sgen['min_min_p_mw'] = 0
        net.gen['controllable'] = True

        # TODO: Currently Ignore reactive power completely (otherwise pp OPF fails)
        cos_phi = 1.0  # This should be a variable in the future?
        for unit_type in ('gen', 'sgen'):
            net[unit_type]['max_s_mva'] = net[unit_type]['max_max_p_mw'] / cos_phi
            net[unit_type]['max_max_q_mvar'] = (
                net[unit_type]['max_s_mva']**2
                - net[unit_type]['max_max_p_mw']**2)**0.5
            net[unit_type]['max_q_mvar'] = net[unit_type]['max_max_q_mvar']
            net[unit_type]['min_q_mvar'] = -net[unit_type]['max_max_q_mvar']

        # TODO: Omit this feature short-term
        self.sgen_idxs = net.sgen.index
        self.gen_idxs = net.gen.index
        # if not n_agents:
        #     self.sgen_idxs = net.sgen.index[net.sgen.p_mw >= min_power]
        #     self.gen_idxs = net.gen.index[net.gen.p_mw >= min_power]
        # else:
        #     if len(net.gen.index) != 0:
        #         self.gen_idxs = np.array(
        #             np.argsort(net.gen.max_p_mw)[::-1][:n_agents])
        #         self.sgen_idxs = np.array([])
        #     else:
        #         self.gen_idxs = np.array([])
        #         self.sgen_idxs = np.array(
        #             np.argsort(net.sgen.max_p_mw)[::-1][:n_agents])

        # assert (len(self.sgen_idxs) + len(self.gen_idxs)) > 0, 'No generators!'

        # Add price params to the network (as poly cost so that the OPF works)
        # Note that the external grids are seen as normal power plants
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=0)
        for idx in self.sgen_idxs:
            pp.create_poly_cost(net, idx, 'sgen', cp1_eur_per_mw=0)
        for idx in self.gen_idxs:
            pp.create_poly_cost(net, idx, 'gen', cp1_eur_per_mw=0)

        net.poly_cost['min_cp1_eur_per_mw'] = 0
        net.poly_cost['max_cp1_eur_per_mw'] = self.max_price

        return net

    def _sampling(self, *args, **kwargs):
        super()._sampling(*args, **kwargs)

        # Sample prices uniformly from min/max range for gens/sgens/ext_grids
        self._sample_from_range(
            'poly_cost', 'cp1_eur_per_mw', self.net.poly_cost.index)

    # def calc_objective(self, net):
        # TODO: There seems to be a slight difference in RL and OPF objective!
        # -> "p_mw[p_mw < 0] = 0.0" is not considered for OPF?!
        """ Minimize costs for active power in the system. """
        # p_mw = net.res_ext_grid['p_mw'].to_numpy().copy()
        # p_mw[p_mw < 0] = 0.0
        # p_mw = np.append(
        #     p_mw, net.res_sgen.p_mw.loc[self.sgen_idxs].to_numpy())
        # p_mw = np.append(p_mw, net.res_gen.p_mw.loc[self.gen_idxs].to_numpy())

        # prices = np.array(net.poly_cost['cp1_eur_per_mw'])

        # assert len(prices) == len(p_mw)

        # # /10000, because too high otherwise
        # return -(np.array(p_mw) * prices).sum() / 10000

if __name__ == '__main__':
    env = EcoDispatch()
    print('EcoDispatch environment created')
    print('Observation space:', env.observation_space.shape)
    print('Action space:', env.action_space.shape)
