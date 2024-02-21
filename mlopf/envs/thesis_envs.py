""" Collection of Reinforcement Learning environments for bachelor and master
thesis experiments. The goal is always to train an agent to learn some kind
of Optimal Power Flow (OPF) calculation.
All these envs can also be solved with
the pandapower OPF to calculate the performance of the DRL agents.

"""

import gym
import numpy as np
import pandapower as pp

from mlopf import opf_env
from mlopf.objectives import min_p_loss, min_pp_costs
from mlopf.build_simbench_net import build_simbench_net

# TODO: Create functions for recurring code (or method in upper class?!)
# TODO: Maybe add one with controllable loads (solvable) and/or storage systems (not solvable with OPF!)
# Needs to be large-scale with big obs/act spaces! (otherwise to easy to solve)
# New kind of objective? eg min losses, min load reduction, TODO
# New kinds of constraints? TODO possibilities?
# Add normal gens to actuators? -> EHV system?!
# Use noise observations? -> stochastic environment! # TODO: simply add this to base env as flag!
# Use multi-level grid? Is this maybe already possible?! # TODO

# TODO: maybe add another env with discrete action spaces (or even both?! but probably separate idea and paper)


class SimpleOpfEnv(opf_env.OpfEnv):
    """
    Standard Optimal Power Flow environment: The grid operator learns to set
    active and reactive power of all generators in the system to maximize
    active power feed-in to the external grid.
    Since this environment has lots of actuators and a
    simple objective function, it is well suited to investigate constraint
    satisfaction.

    Actuators: Active/reactive power of all generators

    Sensors: active+reactive power of all loads; max active power of all gens

    Objective: maximize active power feed-in to external grid

    Constraints: Voltage band, line/trafo load, min/max reactive power,
        constrained reactive power flow over slack bus
    """

    def __init__(self, simbench_network_name='1-LV-rural3--0-sw',
                 gen_scaling=2.5, load_scaling=2.0, cos_phi=0.9,
                 max_q_exchange=0.01, seed=None,
                 *args, **kwargs):

        self.cos_phi = cos_phi
        self.max_q_exchange = max_q_exchange
        self.net = self._define_opf(
            simbench_network_name, gen_scaling=gen_scaling,
            load_scaling=load_scaling, *args, **kwargs)

        # Define the RL problem
        # See all load power values, sgen max active power...
        self.obs_keys = [('sgen', 'max_p_mw', self.net['sgen'].index),
                         ('load', 'p_mw', self.net['load'].index),
                         ('load', 'q_mvar', self.net['load'].index)]

        # ... and control all sgens' active and reactive power values
        self.act_keys = [('sgen', 'p_mw', self.net['sgen'].index),
                         ('sgen', 'q_mvar', self.net['sgen'].index)]
        n_gens = len(self.net['sgen'].index)

        if 'ext_grid_pen_kwargs' not in kwargs:
            kwargs['ext_grid_pen_kwargs'] = {'linear_penalty': 500}
        super().__init__(seed=seed, *args, **kwargs)

        if self.vector_reward is True:
            # 5 penalties and one objective function
            self.reward_space = gym.spaces.Box(
                low=-np.ones(6) * np.inf, high=np.ones(6) * np.inf, seed=seed)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        net.load['controllable'] = False
        net.sgen['controllable'] = True

        net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / self.cos_phi
        # Assumption: Mandatory reactive power provision of cos_phi
        net.sgen['max_max_q_mvar'] = (
            net.sgen['max_s_mva']**2 - net.sgen['max_max_p_mw']**2)**0.5
        net.sgen['max_q_mvar'] = net.sgen['max_max_q_mvar']
        net.sgen['min_q_mvar'] = -net.sgen['max_max_q_mvar']

        # TODO: Currently finetuned for simbench grids '1-LV-urban6--0-sw' and '1-LV-rural3--0-sw'
        net.ext_grid['max_q_mvar'] = self.max_q_exchange
        net.ext_grid['min_q_mvar'] = -self.max_q_exchange

        # OPF objective: Maximize active power feed-in to external grid
        # TODO: Maybe allow for gens here, if necessary
        assert len(net.gen) == 0
        self.active_power_costs = 30
        for idx in net['ext_grid'].index:
            pp.create_poly_cost(net, idx, 'ext_grid',
                                cp1_eur_per_mw=self.active_power_costs)

        return net

    def _sampling(self, step, test, sample_new, *args, **kwargs):
        super()._sampling(step, test, sample_new, *args, **kwargs)

        # Set constraints of current time step (also required for OPF)
        self.net.sgen['max_p_mw'] = self.net.sgen.p_mw * self.net.sgen.scaling


class QMarketEnv(opf_env.OpfEnv):
    """
    Voltage control / Reactive power market environment: The grid operator 
    procures reactive power from generators to minimize losses within its 
    system. 

    Actuators: Reactive power of all gens

    Sensors: active+reactive power of all loads; active power of all gens;
        reactive prices of all gens

    Objective: minimize reactive power costs + minimize loss costs

    Constraints: Voltage band, line/trafo load, min/max reactive power,
        constrained reactive power flow over slack bus

    """

    def __init__(self, simbench_network_name='1-LV-urban6--0-sw',
                 gen_scaling=2.0, load_scaling=1.5, seed=None, min_obs=False,
                 cos_phi=0.9, max_q_exchange=0.01, *args, **kwargs):

        self.cos_phi = cos_phi
        self.max_q_exchange = max_q_exchange
        self.net = self._define_opf(
            simbench_network_name, gen_scaling=gen_scaling,
            load_scaling=load_scaling, *args, **kwargs)

        # Define the RL problem
        # See all load power values, sgen active power, and sgen prices...
        self.obs_keys = [
            ('sgen', 'p_mw', self.net['sgen'].index),
            ('load', 'p_mw', self.net['load'].index),
            ('load', 'q_mvar', self.net['load'].index),
            ('poly_cost', 'cq2_eur_per_mvar2', np.arange(len(self.net.sgen) + len(self.net.ext_grid)))]

        # ... and control all sgens' reactive power values
        self.act_keys = [('sgen', 'q_mvar', self.net['sgen'].index)]

        if 'ext_grid_pen_kwargs' not in kwargs:
            kwargs['ext_grid_pen_kwargs'] = {'linear_penalty': 500}
        super().__init__(seed=seed, *args, **kwargs)

        if self.vector_reward is True:
            # 2 penalties and `n_sgen+1` objective functions
            n_objs = 2 + len(self.net.sgen) + 1
            self.reward_space = gym.spaces.Box(
                low=-np.ones(n_objs) * np.inf, high=np.ones(n_objs) * np.inf, seed=seed)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        net.load['controllable'] = False

        net.sgen['controllable'] = True
        net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / self.cos_phi
        net.sgen['max_max_q_mvar'] = net.sgen['max_s_mva']
        net.sgen['min_min_q_mvar'] = -net.sgen['max_s_mva']

        # TODO: Currently finetuned for simbench grid '1-LV-urban6--0-sw'
        net.ext_grid['max_q_mvar'] = self.max_q_exchange
        net.ext_grid['min_q_mvar'] = -self.max_q_exchange

        # Add price params to the network (as poly cost so that the OPF works)
        # Add loss costs at slack so that objective = loss minimization
        self.loss_costs = 30
        for idx in net.sgen.index:
            pp.create_poly_cost(net, idx, 'sgen',
                                cp1_eur_per_mw=self.loss_costs,
                                cq2_eur_per_mvar2=0)

        for idx in net['ext_grid'].index:
            pp.create_poly_cost(net, idx, 'ext_grid',
                                cp1_eur_per_mw=self.loss_costs,
                                cq2_eur_per_mvar2=0)

        # Load costs are fixed anyway. Added only for completeness.
        for idx in net['load'].index:
            pp.create_poly_cost(net, idx, 'load',
                                cp1_eur_per_mw=-self.loss_costs)

        assert len(net.gen) == 0  # TODO: Maybe add gens here, if necessary

        # Define range from which to sample reactive power prices on market
        self.max_price = 30000
        net.poly_cost['min_cq2_eur_per_mvar2'] = 0
        net.poly_cost['max_cq2_eur_per_mvar2'] = self.max_price

        return net

    def _sampling(self, step, test, sample_new, *args, **kwargs):
        super()._sampling(step, test, sample_new, *args, **kwargs)

        # Sample prices uniformly from min/max range
        self._sample_from_range(
            'poly_cost', 'cq2_eur_per_mvar2',
            self.net.poly_cost[self.net.poly_cost.et == 'sgen'].index)
        self._sample_from_range(
            'poly_cost', 'cq2_eur_per_mvar2',
            self.net.poly_cost[self.net.poly_cost.et == 'ext_grid'].index)

        # active power is not controllable (only relevant for actual OPF)
        self.net.sgen['max_p_mw'] = self.net.sgen.p_mw * self.net.sgen.scaling
        self.net.sgen['min_p_mw'] = 0.999999 * self.net.sgen.max_p_mw

        q_max = (self.net.sgen.max_s_mva**2 - self.net.sgen.max_p_mw**2)**0.5
        self.net.sgen['min_q_mvar'] = -q_max
        self.net.sgen['max_q_mvar'] = q_max

    def calc_objective(self, net):
        """ Define what to do in vector_reward-case. """
        objs = super().calc_objective(net)
        if self.vector_reward:
            # Structure: [sgen1_costs, sgen2_costs, ..., loss_costs]
            return np.append(objs[0:len(self.net.sgen)],
                             sum(objs[len(self.net.sgen):]))
        else:
            return objs

    def calc_penalty(self):
        """ Define what to do in vector_reward-case. """
        # Attention: This probably works only for the default system '1-LV-urban6--0-sw'
        # because only ext_grid q violations there and nothing else
        penalties, valids = super().calc_penalty()
        if self.vector_reward:
            # Structure: [ext_grid_pen, other_pens]
            penalties = np.array((penalties[3], sum(penalties) - penalties[3]))
            valids = np.append(valids[3], np.append(
                valids[0:3], valids[4:]).all())

        return penalties, valids


class EcoDispatchEnv(opf_env.OpfEnv):
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
            kwargs['line_pen_kwargs'] = {'linear_penalty': 100000}
        if 'trafo_pen_kwargs' not in kwargs:
            kwargs['trafo_pen_kwargs'] = {'linear_penalty': 100000}
        if 'ext_grid_pen_kwargs' not in kwargs:
            kwargs['ext_grid_pen_kwargs'] = {'linear_penalty': 10000}
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

    def _sampling(self, step, test, sample_new, *args, **kwargs):
        super()._sampling(step, test, sample_new, *args, **kwargs)

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
