import pdb

import gym
import numpy as np
import pandapower.networks as pn
import pandapower as pp
import simbench as sb

from . import opf_env
from .objectives import (
    max_p_feedin, min_p_loss, add_min_loss_costs, add_max_p_feedin)

""" TODO: Create more examples and use as benchmark later
- Multi-step problems (storage systems as actuators? Ramps? Controllable loads?)
- Not fully observable (some measurements missing) or measurements that get
influenced by actions (eg. voltages)
- Other actuators: controllable loads, storage systems
"""


def min_example():
    """ Some standard non-economic single-step OPF:

    Actuators: active and reactive power of all gens
    Sensors: active and reactive power of all loads; active power limit of gens
    Objective: maximize feed-in (min p reduction)
    Constraints: Voltage band, line/trafo load, apparent power,
        min/max active/reactive power
    """
    net = standard_net()

    # Set the unit constraints...
    # for sampling and...
    net.load['max_p_mw'] = net.load['p_mw'] * 1.0
    net.load['min_p_mw'] = net.load['p_mw'] * 0.05
    net.load['max_q_mvar'] = net.load['max_p_mw'] * 0.3
    net.load['min_q_mvar'] = net.load['min_p_mw'] * 0.3
    net.sgen['max_max_p_mw'] = net.sgen['p_mw'] * 1.0  # technical limit
    net.sgen['min_max_p_mw'] = 0.0
    # ...for actions
    net.sgen['max_p_mw'] = net.sgen['p_mw'] * 1.0  # wind/solar limit
    net.sgen['min_p_mw'] = np.zeros(len(net.sgen.index))
    net.sgen['max_s_mva'] = net.sgen['max_p_mw'] / 0.95  # = cos phi
    net.sgen['max_q_mvar'] = net.sgen['max_s_mva']
    net.sgen['min_q_mvar'] = -net.sgen['max_s_mva']

    # Define the RL problem
    # See all loads and max generator active power...
    obs_keys = [('load', 'p_mw', net['load'].index),
                ('load', 'q_mvar', net['load'].index),
                ('sgen', 'max_p_mw', net['sgen'].index)]
    # TODO: Use relative observations instead?!
    obs_space = get_obs_space(net, obs_keys)

    # ... control all sgens (everything else assumed to be constant)
    act_keys = [('sgen', 'p_mw', net['sgen'].index),
                ('sgen', 'q_mvar', net['sgen'].index)]
    low = np.append(np.zeros(len(net['sgen'].index)),
                    -np.ones(len(net['sgen'].index)))
    high = np.ones(2 * len(net['sgen'].index))
    act_space = gym.spaces.Box(low, high)

    add_max_p_feedin(net)  # Add cost function for OPF

    def objective(net):
        """ Formulate objective function as maximization problem """
        return -max_p_feedin(net)

    pp.runpp(net)

    # Create the environment
    env = opf_env.OpfEnv(net, objective, obs_keys,
                         obs_space, act_keys, act_space)

    return env


def market_example(loss_min=True):
    """ Example how to include power prices into the learning problem.
    Attention: OPF does not work here as reference!!!

    Actuators: active and reactive power of all gens
    Sensors: active+reactive power of all loads;
        active (negative)+reactive(positive) prices of all gens
        TODO: max active power of gens
    Objective: minimize costs (buy reactive power to prevent active power
        reduction)
    Constraints: Voltage band, line/trafo load, apparent power,
        min/max active/reactive power
    """
    net = standard_net()

    # Set the unit constraints...
    # for sampling and
    net.load['max_p_mw'] = net.load['p_mw'] * 1.0
    net.load['min_p_mw'] = net.load['p_mw'] * 0.05
    net.load['max_q_mvar'] = net.load['max_p_mw'] * 0.3
    net.load['min_q_mvar'] = net.load['min_p_mw'] * 0.3
    # ...for actions
    net.sgen['controllable'] = True
    net.sgen['max_p_mw'] = net.sgen['p_mw'] * 1.0
    net.sgen['min_p_mw'] = np.zeros(len(net.sgen.index))
    net.sgen['max_s_mva'] = net.sgen['max_p_mw'] / 0.95  # = cos phi
    net.sgen['max_q_mvar'] = net.sgen['max_s_mva']
    net.sgen['min_q_mvar'] = -net.sgen['max_s_mva']
    # TODO: Problem! This does not work with the OPF because S_max not considered

    # Add price params to the network (as poly cost so that the OPF works)
    for idx in net.sgen.index:
        pp.create_poly_cost(net, idx, 'sgen',
                            cp1_eur_per_mw=1, cq2_eur_per_mvar2=0)
    net.poly_cost['min_cp1_eur_per_mw'] = -30
    net.poly_cost['max_cp1_eur_per_mw'] = 0
    net.poly_cost['min_cq2_eur_per_mvar2'] = 0
    net.poly_cost['max_cq2_eur_per_mvar2'] = 5

    if loss_min:
        # Add loss minimization as another objective
        add_min_loss_costs(net)

    # Define the RL problem
    # See all load power values and sgen prices...
    obs_keys = [('load', 'p_mw', net['load'].index),
                ('load', 'q_mvar', net['load'].index),
                ('poly_cost', 'cp1_eur_per_mw', net['sgen'].index),
                ('poly_cost', 'cq2_eur_per_mvar2', net['sgen'].index)]
    obs_space = get_obs_space(net, obs_keys)

    # ... and control all sgens (everything else assumed to be constant)
    act_keys = [('sgen', 'p_mw', net['sgen'].index),
                ('sgen', 'q_mvar', net['sgen'].index)]
    low = np.append(np.zeros(len(net['sgen'].index)),
                    -np.ones(len(net['sgen'].index)))
    high = np.ones(2 * len(net['sgen'].index))
    act_space = gym.spaces.Box(low, high)

    def objective(net):
        """ Consider quadratic reactive power costs and linear active costs """
        q_costs = (net.poly_cost['cq2_eur_per_mvar2']
                   * net.sgen['q_mvar']**2).sum()
        p_costs = (net.poly_cost['cp1_eur_per_mw'] * net.sgen['p_mw']).sum()
        if loss_min:
            # Grid operator also wants to minimize network active power losses
            loss_costs = min_p_loss(net) * 30  # Assumption: 30€/MWh
        else:
            loss_costs = 0
        return -(q_costs + p_costs + loss_costs)

    pp.runpp(net)

    # Create the environment
    env = opf_env.OpfEnv(net, objective, obs_keys,
                         obs_space, act_keys, act_space)

    return env


def diss_qmarket(loss_min=True, simbench=True):
    """ Minimum working example for the OS2. Should be solvable with OPF for
    comparison. Should be as simple as possible. Should be a market.

    -> reactive power market?!
    Actuators: Reactive power of all gens
    Sensors: active+reactive power of all loads; active power of all gens;
        reactive prices of all gens
    Objective: minimize costs + minimize losses
    Constraints: Voltage band, line/trafo load, min/max reactive power
    TODO: No reactive flow over slack bus as constraint/objective?! (or costs there?)
    -> Yes, this would enforce reactive power demand from grid operator

    (-> re-dispatch market?!
        Actuators: active power of all gens
        Sensors: active+reactive power of all loads; active power of all gens;
            active power prices of all gens
        Objective: minimize costs
        Constraints: Voltage band, line/trafo load, min/max reactive power;
            active power sum = const (-> slack power = const)
        # TODO: sampling: Scenario makes sense only, if there are solvable violations
        # TODO: How to deal with reactive power)
    """
    net = standard_net(simbench)

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
    net.sgen['max_s_mva'] = net.sgen['max_max_p_mw'] / 0.95  # = cos phi

    # Add price params to the network (as poly cost so that the OPF works)
    for idx in net.sgen.index:
        pp.create_poly_cost(net, idx, 'sgen',
                            cp1_eur_per_mw=0, cq2_eur_per_mvar2=0)
    net.poly_cost['min_cq2_eur_per_mvar2'] = 0
    net.poly_cost['max_cq2_eur_per_mvar2'] = 500

    if loss_min:
        # Add loss minimization as another objective
        add_min_loss_costs(net)

    # Define the RL problem
    # See all load power values and sgen prices...
    obs_keys = [('sgen', 'max_p_mw', net['sgen'].index),
                ('load', 'p_mw', net['load'].index),
                ('load', 'q_mvar', net['load'].index),]
                # ('poly_cost', 'cq2_eur_per_mvar2', net['sgen'].index)]
    obs_space = get_obs_space(net, obs_keys)

    # ... and control all sgens (everything else assumed to be constant)
    act_keys = [('sgen', 'q_mvar', net['sgen'].index)]
    low = -np.ones(len(net['sgen'].index))
    high = np.ones(len(net['sgen'].index))
    act_space = gym.spaces.Box(low, high)

    def objective(net):
        """ Consider quadratic reactive power costs and linear active costs """
        q_costs = (net.poly_cost['cq2_eur_per_mvar2']
                   * net.sgen['q_mvar']**2).sum()
        if loss_min:
            # Grid operator also wants to minimize network active power losses
            loss_costs = min_p_loss(net) * 30  # Assumption: 30€/MWh
        else:
            loss_costs = 0
        return -(q_costs + loss_costs)

    def sampling(env):
        for unit_type, column, idxs in env.sample_keys:
            low = env.net[unit_type][f'min_{column}'].loc[idxs]
            high = env.net[unit_type][f'max_{column}'].loc[idxs]
            r = np.random.uniform(low, high, size=(len(idxs),))
            env.net[unit_type][column].loc[idxs] = r
        # active power is not controllable (only relevant for OPF)
        env.net.sgen['min_p_mw'] = env.net.sgen['max_p_mw']
        env.net.sgen['p_mw'] = env.net.sgen['max_p_mw']
        q_max = (env.net.sgen['max_s_mva']**2 - env.net.sgen['p_mw']**2)**0.5
        env.net.sgen['min_q_mvar'] = -q_max
        env.net.sgen['max_q_mvar'] = q_max

    pp.runpp(net)

    # Create the environment
    env = opf_env.OpfEnv(net, objective, obs_keys,
                         obs_space, act_keys, act_space, sampling=sampling)

    return env


def standard_net(simbench=False):
    # Create a power system and set some constraints
    if simbench:
        net = sb.get_simbench_net('1-LV-urban6--0-sw')
        # '1-LV-rural1--0-sw')  # 4 sgen
        # 1-LV-urban6--0-sw # 5 sgen
        # 1-LV-rural2--0-sw # 8 sgen
        # Scale up loads and gens
        net.sgen['scaling'] = 1.5
        net.load['scaling'] = 1.5
    else:
        net = pn.create_cigre_network_mv(with_der="pv_wind")

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
    env = min_example()
    obs = env.reset()
    for _ in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        print(reward)
        print(obs)
        print(info['penalty'])
        print(env.net.res_bus)
        print('')
        if done:
            obs = env.reset()
