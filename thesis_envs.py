""" Collection of Reinforcement Learning environments for bachelor and master
thesis experiments. The goal is always to train an agent to learn some kind
of Optimal Power Flow (OPF) calculation. """

import gym
import numpy as np
import pandapower as pp
import simbench as sb

from . import opf_env
from .objectives import (min_p_loss, add_min_loss_costs)


class QMarketEnv(opf_env.OpfEnv):
    """
    Reactive power market environment: The grid operator procures reactive power
    from generators to minimize losses within its system.

    Actuators: Reactive power of all gens

    Sensors: active+reactive power of all loads; active power of all gens;
        reactive prices of all gens

    Objective: minimize reactive power costs + minimize loss costs

    Constraints: Voltage band, line/trafo load, min/max reactive power, zero
        reactive power flow over slack bus

    """

    def __init__(self, simbench_network_name='small',
                 multi_agent_case=False):
        self.multi_agent_case = multi_agent_case
        self.net = self._build_net(simbench_network_name)

        # Define the RL problem
        # See all load power values, sgen active power, and sgen prices...
        self.obs_keys = [('sgen', 'max_p_mw', self.net['sgen'].index),
                         ('load', 'p_mw', self.net['load'].index),
                         ('load', 'q_mvar', self.net['load'].index)]
        if not multi_agent_case:
            # In the multi-agent case, other learning agents would provide the bids
            self.obs_keys.append(
                ('poly_cost', 'cq2_eur_per_mvar2', self.net['sgen'].index))
        self.observation_space = get_obs_space(self.net, self.obs_keys)

        # ... and control all sgens' reactive power values
        self.act_keys = [('sgen', 'q_mvar', self.net['sgen'].index)]
        low = -np.ones(len(self.net['sgen'].index))
        high = np.ones(len(self.net['sgen'].index))
        self.action_space = gym.spaces.Box(low, high)

        super().__init__()

    def _build_net(self, simbench_network_name):
        net, self.profiles = build_net(simbench_network_name)

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

        # TODO: Make sure everything is correctly scaled
        net.sgen['max_max_p_mw'] = self.profiles[('sgen', 'p_mw')].max(
            axis=0) * net['sgen']['scaling']
        net.sgen['min_max_p_mw'] = 0
        net.sgen['controllable'] = True
        cos_phi = 0.9
        net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / cos_phi
        # TODO: THis strange /scaling should not be here, but maybe in apply_actions instead?!
        net.sgen['max_max_q_mvar'] = net.sgen['max_s_mva'] / net.sgen.scaling

        # TODO: Stand jetzt abgestimmt für Netz '1-LV-urban6--0-sw'
        net.ext_grid['max_q_mvar'] = 0.05
        net.ext_grid['min_q_mvar'] = -0.05
        # TODO: is scaling correctly considered here? (test by looking at OPF results -> should be these values here!)

        # Add price params to the network (as poly cost so that the OPF works)
        self.loss_costs = 30
        for idx in net.sgen.index:
            pp.create_poly_cost(net, idx, 'sgen',
                                cp1_eur_per_mw=self.loss_costs,
                                cq2_eur_per_mvar2=0)
        assert len(net.gen) == 0  # Maybe add gens if necessary here
        for idx in net['ext_grid'].index:
            pp.create_poly_cost(net, idx, 'ext_grid',
                                cp1_eur_per_mw=self.loss_costs)
        # Define range from which to sample reactive power prices on market
        net.poly_cost['min_cq2_eur_per_mvar2'] = 0
        net.poly_cost['max_cq2_eur_per_mvar2'] = 10000

        pp.runpp(net)

        return net

    def _sampling(self):
        """ Assumption: Only simbench systems with timeseries data are used. """
        self._set_simbench_state()

        # Sample prices uniformly from min/max range
        if not self.multi_agent_case:
            self._sample_from_range( # TODO: Are the indexes here correct??
                'poly_cost', 'cq2_eur_per_mvar2', self.net['sgen'].index)

        # active power is not controllable (only relevant for actual OPF)
        self.net.sgen['max_p_mw'] = self.net.sgen['p_mw'] * self.net.sgen['scaling']
        self.net.sgen['min_p_mw'] = 0.9999 * self.net.sgen['max_p_mw']

        q_max = (self.net.sgen['max_s_mva']**2 - (self.net.sgen['p_mw']*self.net.sgen['scaling'])**2)**0.5
        self.net.sgen['min_q_mvar'] = -q_max
        self.net.sgen['max_q_mvar'] = q_max

    def _calc_reward(self, net):
        """ Consider quadratic reactive power costs on the market and linear
        active costs for losses in the system. """
        if self.multi_agent_case:
            # The agents handle their trading internally here
            q_costs = 0
        else:
            q_costs = (net.poly_cost['cq2_eur_per_mvar2'].loc[net.sgen.index]
                       * net.res_sgen['q_mvar']**2).sum()

        # Grid operator also wants to minimize network active power losses
        loss_costs = min_p_loss(net) * self.loss_costs


        # print('Reward distr: ', q_costs, loss_costs)  # for testing

        return -q_costs - loss_costs


class EcoDispatchEnv(opf_env.OpfEnv):
    # TODO: Maybe as replacement for qmarket?!
    pass


def build_net(simbench_network_name='small'):
    """ Init and return a simbench power network with standard configuration.
    """

    # Choose one of the standard cases
    if simbench_network_name == 'small':
        # TODO: Decide which ones to actually use (small should mean small obs and act space!!!)
        net = sb.get_simbench_net('1-LV-urban6--0-sw')
    elif simbench_network_name == 'medium':
        net = sb.get_simbench_net('1-HV-mixed--0-sw')
    elif simbench_network_name == 'large':
        net = sb.get_simbench_net('1-MV-urban--0-sw')
    else:
        # No standard case was selected
        net = sb.get_simbench_net(simbench_network_name)

    # Scale up loads and gens to make task a bit more difficult
    # (TODO: Maybe requires fine-tuning)
    net.sgen['scaling'] = 2.0
    net.load['scaling'] = 1.5

    # Set the system constraints
    # Define the voltage band of +-5%
    net.bus['max_vm_pu'] = 1.05
    net.bus['min_vm_pu'] = 0.95
    # Set maximum loading of lines and transformers
    net.line['max_loading_percent'] = 80
    net.trafo['max_loading_percent'] = 80

    assert not sb.profiles_are_missing(net)
    profiles = sb.get_absolute_values(net,
                                      profiles_instead_of_study_cases=True)

    return net, profiles


def get_obs_space(net, obs_keys: list):
    lows, highs = [], []
    for unit_type, column, idxs in obs_keys:
        lows.append(net[unit_type][f'min_{column}'].loc[idxs])
        highs.append(net[unit_type][f'max_{column}'].loc[idxs])

    return gym.spaces.Box(
        np.concatenate(lows, axis=0), np.concatenate(highs, axis=0))


if __name__ == '__main__':
    env = qmarket_env()
    obs = env.reset()
    for _ in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        print('Reward: ', reward)
        print('Penalty (constraints): ', info['penalty'])
        print('Observation: ', obs)
        print('')
        if done:
            obs = env.reset()
