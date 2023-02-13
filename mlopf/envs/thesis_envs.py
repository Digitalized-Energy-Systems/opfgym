""" Collection of Reinforcement Learning environments for bachelor and master
thesis experiments. The goal is always to train an agent to learn some kind
of Optimal Power Flow (OPF) calculation.
All these envs can also be solved with
the pandapower OPF to calculate the performance of the DRL agents.

"""

import gym
import numpy as np
import pandapower as pp
import simbench as sb

from mlopf import opf_env
from mlopf.objectives import min_p_loss
from mlopf.penalties import (ext_grid_overpower, active_reactive_overpower)

# TODO: Create functions for recurring code (or method in upper class?!)
# TODO: Maybe add one with controllable loads (solvable) and/or storage systems (not solvable with OPF!)


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

    Constraints: Voltage band, line/trafo load, min/max reactive power, zero
        reactive power flow over slack bus
    """

    def __init__(self, simbench_network_name='1-LV-rural3--0-sw', gen_scaling=2.0, load_scaling=2.0, seed=None, *args, **kwargs):
        self.net = self._build_net(
            simbench_network_name, gen_scaling, load_scaling)

        # Define the RL problem
        # See all load power values, sgen max active power...
        self.obs_keys = [('sgen', 'max_p_mw', self.net['sgen'].index),
                         ('load', 'p_mw', self.net['load'].index),
                         ('load', 'q_mvar', self.net['load'].index)]

        # ... and control all sgens' active and reactive power values
        self.act_keys = [('sgen', 'p_mw', self.net['sgen'].index),
                         ('sgen', 'q_mvar', self.net['sgen'].index)]
        n_gens = len(self.net['sgen'].index)
        low = np.concatenate([np.zeros(n_gens), -np.ones(n_gens)])
        high = np.ones(2 * n_gens)
        self.action_space = gym.spaces.Box(low, high, seed=seed)

        super().__init__(apparent_power_penalty=5000, seed=seed, *args, **kwargs)

        if self.vector_reward is True:
            # 5 penalties and one objective function
            self.reward_space = gym.spaces.Box(
                low=-np.ones(6) * np.inf, high=np.ones(6) * np.inf, seed=seed)

    def _build_net(self, simbench_network_name, gen_scaling, load_scaling):
        net, self.profiles = build_net(
            simbench_network_name, gen_scaling, load_scaling)

        net.load['controllable'] = False
        # Constraints required for observation space only
        net.load['min_p_mw'] = self.profiles[('load', 'p_mw')].min(
            axis=0) * net['load']['scaling']
        net.load['max_p_mw'] = self.profiles[('load', 'p_mw')].max(
            axis=0) * net['load']['scaling']
        net.load['min_q_mvar'] = self.profiles[('load', 'q_mvar')].min(
            axis=0) * net['load']['scaling']
        net.load['max_q_mvar'] = self.profiles[('load', 'q_mvar')].max(
            axis=0) * net['load']['scaling']  # TODO: This code repeats often

        net.sgen['max_max_p_mw'] = self.profiles[('sgen', 'p_mw')].max(
            axis=0) * net['sgen']['scaling']
        net.sgen['min_max_p_mw'] = self.profiles[('sgen', 'p_mw')].min(
            axis=0) * net['sgen']['scaling']

        # Some power values are always zero (for whatever reason?!)
        # TODO: Do this in base class?
        net.sgen.drop(
            net.sgen[net.sgen.max_max_p_mw == 0.0].index, inplace=True)
        net.load.drop(
            net.load[net.load.min_q_mvar == net.load.max_q_mvar].index, inplace=True)
        net.load.drop(
            net.load[net.load.min_p_mw == net.load.max_p_mw].index, inplace=True)

        net.sgen['controllable'] = True

        cos_phi = 0.9
        net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / cos_phi
        # Assumption: Mandatory reactive power provision of cos_phi
        net.sgen['max_max_q_mvar'] = (
            net.sgen['max_s_mva']**2 - net.sgen['max_max_p_mw']**2)**0.5
        net.sgen['max_q_mvar'] = net.sgen['max_max_q_mvar']
        net.sgen['min_q_mvar'] = -net.sgen['max_max_q_mvar']

        # TODO: Currently finetuned for simbench grids '1-LV-urban6--0-sw' and '1-LV-rural3--0-sw'
        net.ext_grid['max_q_mvar'] = 0.01
        net.ext_grid['min_q_mvar'] = -0.01

        # OPF objective: Maximize active power feed-in to external grid
        # TODO: Maybe allow for gens here, if necessary
        assert len(net.gen) == 0
        self.active_power_costs = 30
        for idx in net['ext_grid'].index:
            pp.create_poly_cost(net, idx, 'ext_grid',
                                cp1_eur_per_mw=self.active_power_costs)

        return net

    def _sampling(self, step=None, test=False):
        """ Assumption: Only simbench systems with timeseries data are used. """
        self._set_simbench_state(step, test)

        # Set constraints of current time step (also required for OPF)
        self.net.sgen['max_p_mw'] = self.net.sgen.p_mw * self.net.sgen.scaling

    def _calc_reward(self, net):
        """ Objective: Maximize active power feed-in to external grid. """
        return -(self.net.res_ext_grid.p_mw * self.active_power_costs).sum()

    def _calc_penalty(self):
        penalty = super()._calc_penalty()
        # Do not allow for high reactive power exchange with external grid
        penalty.append(-ext_grid_overpower(
            self.net, self.ext_overpower_penalty, 'q_mvar'))

        return penalty


class QMarketEnv(opf_env.OpfEnv):
    """
    Reactive power market environment (base case): The grid operator procures
    reactive power from generators to minimize losses within its system. There
    are also variants of this env where the market participants learn to bid on
    the market or where grid operator and market participants learn at the same
    time.

    Actuators: Reactive power of all gens

    Sensors: active+reactive power of all loads; active power of all gens;
        reactive prices of all gens

    Objective: minimize reactive power costs + minimize loss costs

    Constraints: Voltage band, line/trafo load, min/max reactive power, zero
        reactive power flow over slack bus

    """

    def __init__(self, simbench_network_name='1-LV-urban6--0-sw',
                 gen_scaling=2.0, load_scaling=1.5, seed=None, *args, **kwargs):
        self.net = self._build_net(
            simbench_network_name, gen_scaling, load_scaling)

        # Define the RL problem
        # See all load power values, sgen active power, and sgen prices...
        # TODO: Add current time as observation! (see attack paper)
        self.obs_keys = [
            ('sgen', 'p_mw', self.net['sgen'].index),
            ('load', 'p_mw', self.net['load'].index),
            ('load', 'q_mvar', self.net['load'].index),  # TODO: res_load?!
            ('poly_cost', 'cq2_eur_per_mvar2', self.net.poly_cost.index)]

        # ... and control all sgens' reactive power values
        self.act_keys = [('sgen', 'q_mvar', self.net['sgen'].index)]
        low = -np.ones(len(self.net['sgen'].index))
        high = np.ones(len(self.net['sgen'].index))
        self.action_space = gym.spaces.Box(low, high, seed=seed)

        super().__init__(ext_overpower_penalty=250,
                         apparent_power_penalty=1500,
                         seed=seed,
                         *args, **kwargs)

        if self.vector_reward is True:
            # 4 penalties and one objective function
            self.reward_space = gym.spaces.Box(
                low=-np.ones(5) * np.inf, high=np.ones(5) * np.inf, seed=seed)

    def _build_net(self, simbench_network_name, gen_scaling, load_scaling):
        net, self.profiles = build_net(
            simbench_network_name, gen_scaling, load_scaling)

        net.load['controllable'] = False
        # Constraints required for observation space only
        net.load['min_p_mw'] = self.profiles[('load', 'p_mw')].min(
            axis=0) * net['load']['scaling']
        net.load['max_p_mw'] = self.profiles[('load', 'p_mw')].max(
            axis=0) * net['load']['scaling']
        net.load['min_q_mvar'] = self.profiles[('load', 'q_mvar')].min(
            axis=0) * net['load']['scaling']
        net.load['max_q_mvar'] = self.profiles[('load', 'q_mvar')].max(
            axis=0) * net['load']['scaling']

        net.sgen['max_max_p_mw'] = self.profiles[('sgen', 'p_mw')].max(
            axis=0) * net['sgen']['scaling']
        net.sgen['min_max_p_mw'] = self.profiles[('sgen', 'p_mw')].min(
            axis=0) * net['sgen']['scaling']
        net.sgen['max_p_mw'] = net.sgen['max_max_p_mw']
        net.sgen['min_p_mw'] = net.sgen['min_max_p_mw']

        # Some power values are always zero (for whatever reason?!)
        net.sgen.drop(
            net.sgen[net.sgen.max_max_p_mw == 0.0].index, inplace=True)
        net.load.drop(
            net.load[net.load.min_q_mvar == net.load.max_q_mvar].index, inplace=True)
        net.load.drop(
            net.load[net.load.min_p_mw == net.load.max_p_mw].index, inplace=True)

        net.sgen['controllable'] = True
        cos_phi = 0.90
        net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / cos_phi
        net.sgen['max_max_q_mvar'] = net.sgen['max_s_mva']

        # TODO: Currently finetuned for simbench grid '1-LV-urban6--0-sw'
        # TODO: Maybe see ext grid as just another reactive power provider?! (costs instead of constraints)
        # Advantage: That would remove one hyperparametery
        net.ext_grid['max_q_mvar'] = 0.01
        net.ext_grid['min_q_mvar'] = -0.01  # TODO: verify this

        # Add price params to the network (as poly cost so that the OPF works)
        self.loss_costs = 30
        for idx in net.sgen.index:
            pp.create_poly_cost(net, idx, 'sgen',
                                cp1_eur_per_mw=self.loss_costs,
                                cq2_eur_per_mvar2=0)
        assert len(net.gen) == 0  # TODO: Maybe add gens here, if necessary
        for idx in net['ext_grid'].index:
            pp.create_poly_cost(net, idx, 'ext_grid',
                                cp1_eur_per_mw=self.loss_costs,
                                cq2_eur_per_mvar2=0)  # TODO: Verify this
        # Define range from which to sample reactive power prices on market
        self.max_price = 30000
        net.poly_cost['min_cq2_eur_per_mvar2'] = 0
        net.poly_cost['max_cq2_eur_per_mvar2'] = self.max_price

        return net

    def _sampling(self, step=None, test=False):
        """ Assumption: Only simbench systems with timeseries data are used. """
        self._set_simbench_state(step, test)

        # Sample prices uniformly from min/max range
        self._sample_from_range(  # TODO: Are the indexes here correct??
            'poly_cost', 'cq2_eur_per_mvar2', self.net.poly_cost.index)
        # TODO: Verify this (test for slack as q provider)

        # active power is not controllable (only relevant for actual OPF)
        self.net.sgen['max_p_mw'] = self.net.sgen.p_mw * self.net.sgen.scaling
        self.net.sgen['min_p_mw'] = 0.999999 * self.net.sgen.max_p_mw

        q_max = (self.net.sgen['max_s_mva']**2 -
                 (self.net.sgen.p_mw * self.net.sgen.scaling)**2)**0.5
        self.net.sgen['min_q_mvar'] = -q_max
        self.net.sgen['max_q_mvar'] = q_max

    def _calc_reward(self, net):
        """ Consider quadratic reactive power costs on the market and linear
        active costs for losses in the system. """
        q_costs = (net.poly_cost[net.poly_cost.et == 'sgen'].cq2_eur_per_mvar2
                   * net.res_sgen.q_mvar.to_numpy()**2).sum()
        if (net.poly_cost.et == 'ext_grid').any():
            mask = net.poly_cost.et == 'ext_grid'
            prices = net.poly_cost.cq2_eur_per_mvar2[mask].to_numpy()
            q_mvar = net.res_ext_grid.q_mvar.to_numpy()
            q_costs += sum(prices * q_mvar**2)

        # Grid operator also wants to minimize network active power losses
        loss_costs = min_p_loss(net) * self.loss_costs

        return -q_costs - loss_costs

    def _calc_penalty(self):
        penalty = super()._calc_penalty()
        penalty.append(-ext_grid_overpower(
            self.net, self.ext_overpower_penalty, 'q_mvar'))

        return penalty


class EcoDispatchEnv(opf_env.OpfEnv):
    """
    Economic Dispatch/Active power market environment: The grid operator
    procures active power from generators to minimize losses within its system.

    Actuators: Active power of all gens (reactive power?!)

    Sensors: active+reactive power of all loads; (TODO: active power of all other gens);
        active power prices of all gens

    Objective: minimize active power costs + minimize loss costs

    Constraints: Voltage band, line/trafo load, min/max active power limits (automatically)

    """

    def __init__(self, simbench_network_name='1-HV-urban--0-sw', min_power=0,
                 n_agents=None, gen_scaling=1.0, load_scaling=1.5, u_penalty=300,
                 overload_penalty=10, ext_overpower_penalty=0.01, max_price=600,
                 seed=None,
                 *args, **kwargs):
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

        self.net = self._build_net(
            simbench_network_name, min_power, n_agents, gen_scaling, load_scaling)

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
        self._set_action_space(seed)
        # TODO: Define constraints explicitly?! (active power min/max not default!)

        super().__init__(u_penalty=u_penalty,
                         overload_penalty=overload_penalty,
                         ext_overpower_penalty=ext_overpower_penalty,
                         seed=seed,
                         *args, **kwargs)

    def _set_action_space(self, seed):
        """ Each power plant can be set in range from 0-100% power
        (minimal power higher than zero not considered here) """
        low = np.zeros(len(self.act_keys[0][2]) + len(self.act_keys[1][2]))
        high = np.ones(len(self.act_keys[0][2]) + len(self.act_keys[1][2]))
        self.action_space = gym.spaces.Box(low, high, seed=seed)

    def _build_net(self, simbench_network_name, min_power, n_agents, gen_scaling=1.0, load_scaling=1.5):
        net, self.profiles = build_net(
            simbench_network_name, gen_scaling, load_scaling)
        # Set voltage setpoints a bit higher than 1.0 to consider voltage drop?
        net.ext_grid['vm_pu'] = 1.0
        net.gen['vm_pu'] = 1.0

        net.load['controllable'] = False
        # Load constraints required for observation space only
        net.load['min_p_mw'] = self.profiles[('load', 'p_mw')].min(
            axis=0) * net['load']['scaling']
        net.load['max_p_mw'] = self.profiles[('load', 'p_mw')].max(
            axis=0) * net['load']['scaling']
        net.load['min_q_mvar'] = self.profiles[('load', 'q_mvar')].min(
            axis=0) * net['load']['scaling']
        net.load['max_q_mvar'] = self.profiles[('load', 'q_mvar')].max(
            axis=0) * net['load']['scaling']

        # Generator constraints required for observation and OPF!
        net.sgen['min_p_mw'] = 0
        net.sgen['max_p_mw'] = self.profiles[('sgen', 'p_mw')].max(
            axis=0) * net['sgen']['scaling']
        net.gen['min_p_mw'] = 0
        net.gen['max_p_mw'] = self.profiles[('gen', 'p_mw')].max(
            axis=0) * net['gen']['scaling']
        net.sgen['max_max_p_mw'] = net.sgen['max_p_mw']
        net.gen['max_max_p_mw'] = net.gen['max_p_mw']
        # Some power values are always zero (for whatever reason?!)
        net.sgen.drop(
            net.sgen[net.sgen.max_max_p_mw == 0.0].index, inplace=True)
        net.load.drop(
            net.load[net.load.min_q_mvar == net.load.max_q_mvar].index, inplace=True)
        net.load.drop(
            net.load[net.load.min_p_mw == net.load.max_p_mw].index, inplace=True)

        net.ext_grid['min_p_mw'] = 0

        # TODO: Also for gen
        #     axis=0) * net['sgen']['scaling']
        # net.sgen['min_max_p_mw'] = 0
        net.sgen['controllable'] = True
        net.gen['controllable'] = True

        cos_phi = 1.0
        for unit_type in ('gen', 'sgen'):
            net[unit_type]['max_s_mva'] = net[unit_type]['max_max_p_mw'] / cos_phi
            net[unit_type]['max_max_q_mvar'] = (
                net[unit_type]['max_s_mva']**2
                - net[unit_type]['max_max_p_mw']**2)**0.5
            net[unit_type]['max_q_mvar'] = net[unit_type]['max_max_q_mvar']
            net[unit_type]['min_q_mvar'] = -net[unit_type]['max_max_q_mvar']
            # TODO: Here, probably a better solution is required

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

    def _sampling(self, step=None, test=False):
        """ Assumption: Only simbench systems with timeseries data are used. """
        self._set_simbench_state(step, test)

        # Sample prices uniformly from min/max range for gens/sgens/ext_grids
        self._sample_from_range(
            'poly_cost', 'cp1_eur_per_mw', self.net.poly_cost.index)

    def _calc_reward(self, net):
        """ Minimize costs for active power in the system. """
        p_mw = net.res_ext_grid['p_mw'].to_numpy().copy()
        p_mw[p_mw < 0] = 0.0
        p_mw = np.append(
            p_mw, net.res_sgen.p_mw.loc[self.sgen_idxs].to_numpy())
        p_mw = np.append(p_mw, net.res_gen.p_mw.loc[self.gen_idxs].to_numpy())

        prices = np.array(net.poly_cost['cp1_eur_per_mw'])

        assert len(prices) == len(p_mw)

        # /10000, because too high otherwise
        return -(np.array(p_mw) * prices).sum() / 10000

    def _calc_penalty(self):
        penalty = super()._calc_penalty()
        penalty.append(-ext_grid_overpower(
            self.net,
            penalty_factor=self.ext_overpower_penalty,
            column='p_mw'))
        return penalty


def build_net(simbench_network_name, gen_scaling=1.0, load_scaling=2.0,
              voltage_band=0.05, max_loading=80):
    """ Init and return a simbench power network with standard configuration.
    """

    # No standard case was selected
    net = sb.get_simbench_net(simbench_network_name)

    # Scale up loads to make task a bit more difficult
    # (TODO: Maybe requires fine-tuning and should be done env-wise)
    net.sgen['scaling'] = gen_scaling
    net.gen['scaling'] = gen_scaling
    net.load['scaling'] = load_scaling

    # Set the system constraints
    # Define the voltage band of +-5%
    net.bus['max_vm_pu'] = 1 + voltage_band
    net.bus['min_vm_pu'] = 1 - voltage_band
    # Set maximum loading of lines and transformers
    net.line['max_loading_percent'] = max_loading
    net.trafo['max_loading_percent'] = max_loading

    assert not sb.profiles_are_missing(net)
    profiles = sb.get_absolute_values(net,
                                      profiles_instead_of_study_cases=True)
    # Fix strange error in simbench: Sometimes negative active power values
    profiles[('sgen', 'p_mw')][profiles[('sgen', 'p_mw')] < 0.0] = 0.0

    return net, profiles


if __name__ == '__main__':
    env = SimpleOpfEnv()
    obs = env.reset()
    for _ in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        print('Reward: ', reward)
        print('Penalty (constraints): ', info['penalty'])
        print('Observation: ', obs)
        print('')
        if done:
            obs = env.reset()
