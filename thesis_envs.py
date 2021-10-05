""" Collection of Reinforcement Learning environments for bachelor and master
thesis experiments. The goal is always to train an agent to learn some kind
of Optimal Power Flow (OPF) calculation. """

import gym
import numpy as np
import pandapower.networks as pn
import pandapower as pp
import simbench as sb

from . import opf_env
from .objectives import (min_p_loss, add_min_loss_costs)


def qmarket_env(loss_min=True, simbench_network_name='small',
                multi_agent_case=False):
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

    net = build_net(simbench_network_name=simbench_network_name)

    # TODO ThW: Some of this can be deleted with new sampling method
    # Set the unit constraints...
    # for sampling and
    net.load['max_p_mw'] = net.load['p_mw'] * 1.0
    net.load['min_p_mw'] = net.load['p_mw'] * 0.05
    net.load['max_q_mvar'] = net.load['max_p_mw'] * 0.3
    net.load['min_q_mvar'] = net.load['min_p_mw'] * 0.3
    # ...for actions
    net.sgen['controllable'] = True
    net.sgen['max_p_mw'] = net.sgen['p_mw']
    net.sgen['max_max_p_mw'] = net.sgen['max_p_mw']
    net.sgen['min_max_p_mw'] = np.zeros(len(net.sgen.index))
    net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / 0.90  # = cos phi
    net.sgen['max_max_q_mvar'] = net.sgen['max_s_mva']
    net.sgen['min_max_q_mvar'] = -net.sgen['max_s_mva']

    net.ext_grid['max_q_mvar'] = 0.05  # TODO: Stand jetzt abgestimmt für
    net.ext_grid['min_q_mvar'] = -0.05  # Netz '1-LV-urban6--0-sw'
    # TODO: is scaling correctly considered here? (test by looking at OPF results -> should be these values here!)

    # Add price params to the network (as poly cost so that the OPF works)
    for idx in net.sgen.index:
        pp.create_poly_cost(net, idx, 'sgen',
                            cp1_eur_per_mw=0, cq2_eur_per_mvar2=0)
    net.poly_cost['min_cq2_eur_per_mvar2'] = 0
    net.poly_cost['max_cq2_eur_per_mvar2'] = 10000

    if loss_min:
        # Add loss minimization as another objective
        add_min_loss_costs(net)

    # Define the RL problem
    # See all load power values, sgen active power, and sgen prices...
    obs_keys = [('sgen', 'max_p_mw', net['sgen'].index),
                ('load', 'p_mw', net['load'].index),
                ('load', 'q_mvar', net['load'].index)]
    if not multi_agent_case:
        # In the multi-agent case, other learning agents would provide the bids
        obs_keys.append(('poly_cost', 'cq2_eur_per_mvar2', net['sgen'].index))
    obs_space = get_obs_space(net, obs_keys)

    # ... and control all sgens (everything else assumed to be constant)
    act_keys = [('sgen', 'q_mvar', net['sgen'].index)]
    low = -np.ones(len(net['sgen'].index))
    high = np.ones(len(net['sgen'].index))
    act_space = gym.spaces.Box(low, high)

    def objective(net):
        """ Consider quadratic reactive power costs and linear active costs """
        if multi_agent_case:
            # The agents handle their trading internally here
            q_costs = 0
        else:
            q_costs = (net.poly_cost['cq2_eur_per_mvar2']
                       * net.res_sgen['q_mvar']**2).sum()
        if loss_min:
            # Grid operator also wants to minimize network active power losses
            loss_costs = min_p_loss(net) * 30  # Assumption: 30€/MWh
        else:
            loss_costs = 0

        # print('Reward distr: ', q_costs, loss_costs)  # for testing

        return -q_costs - loss_costs

    def sampling(env):
        # TODO ThW: Sample from simbench time-series data
        # + Sample bids if in single agent case!
        for unit_type, column, idxs in env.sample_keys:
            low = env.net[unit_type][f'min_{column}'].loc[idxs]
            high = env.net[unit_type][f'max_{column}'].loc[idxs]
            r = np.random.uniform(low, high, size=(len(idxs),))
            env.net[unit_type][column].loc[idxs] = r
        # active power is not controllable (only relevant for OPF)
        env.net.sgen['p_mw'] = env.net.sgen['max_p_mw']
        env.net.sgen['min_p_mw'] = 0.9999 * env.net.sgen['p_mw']
        env.net.sgen['max_p_mw'] = env.net.sgen['p_mw']

        q_max = (env.net.sgen['max_s_mva']**2 - env.net.sgen['p_mw']**2)**0.5
        env.net.sgen['min_q_mvar'] = -q_max
        env.net.sgen['max_q_mvar'] = q_max

    pp.runpp(net)

    # Create the environment
    env = opf_env.OpfEnv(net, objective, obs_keys,
                         obs_space, act_keys, act_space, sampling=sampling)

    return env


def build_net(simbench_network_name='small'):
    """ Init and return a simbench power network """

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
    # (Maybe requires fine-tuning)
    net.sgen['scaling'] = 2.0
    net.load['scaling'] = 1.5

    # Set the system constraints
    # Define the voltage band of +-5%
    net.bus['max_vm_pu'] = 1.05
    net.bus['min_vm_pu'] = 0.95
    # Set maximum loading of lines and transformers
    net.line['max_loading_percent'] = 80
    net.trafo['max_loading_percent'] = 80
    return net


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
