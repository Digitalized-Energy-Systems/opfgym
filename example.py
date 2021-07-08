import gym
import numpy as np
import pandapower.networks as pn
import pandapower as pp

from . import opf_env
from .objectives import max_p_feedin

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
    # Create a power system and set some constraints
    net = pn.create_cigre_network_mv(with_der="pv_wind")

    # Set the system constraints
    # Define the voltage band of +-5%
    net.bus['max_vm_pu'] = 1.05
    net.bus['min_vm_pu'] = 0.95
    # Set maximum loading of lines and transformers
    net.line['max_loading_percent'] = 80
    net.trafo['max_loading_percent'] = 80

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

    def objective(net):
        """ Formulate objective function as maximization problem """
        return -max_p_feedin(net)

    pp.runpp(net)

    # Create the environment
    env = opf_env.OpfEnv(net, objective, obs_keys,
                         obs_space, act_keys, act_space)

    return env


def market_example():
    """ Example how to include power prices into the learning problem

    Actuators: active and reactive power of all gens
    Sensors: active+reactive power of all loads;
        active (negative)+reactive(positive) prices of all gens
    Objective: minimize costs (buy reactive power to prevent active power
        reduction)
    Constraints: Voltage band, line/trafo load, apparent power,
        min/max active/reactive power
    """
    # Create a power system and set some constraints
    net = pn.create_cigre_network_mv(with_der="pv_wind")

    # Set the system constraints
    # Define the voltage band of +-5%
    net.bus['max_vm_pu'] = 1.05
    net.bus['min_vm_pu'] = 0.95
    # Set maximum loading of lines and transformers
    net.line['max_loading_percent'] = 80
    net.trafo['max_loading_percent'] = 80

    # Set the unit constraints...
    # for sampling and
    net.load['max_p_mw'] = net.load['p_mw'] * 1.0
    net.load['min_p_mw'] = net.load['p_mw'] * 0.05
    net.load['max_q_mvar'] = net.load['max_p_mw'] * 0.3
    net.load['min_q_mvar'] = net.load['min_p_mw'] * 0.3
    # ...for actions
    net.sgen['max_p_mw'] = net.sgen['p_mw'] * 1.0
    net.sgen['min_p_mw'] = np.zeros(len(net.sgen.index))
    net.sgen['max_s_mva'] = net.sgen['max_p_mw'] / 0.95  # = cos phi
    net.sgen['max_q_mvar'] = net.sgen['max_s_mva']
    net.sgen['min_q_mvar'] = -net.sgen['max_s_mva']

    # Add some params to the network that normally do not exist
    net.sgen['min_p_price'] = -30
    net.sgen['max_p_price'] = 0
    net.sgen['min_q_price'] = 0
    net.sgen['max_q_price'] = 5
    net.sgen['p_price'] = 0  # Init values. Overwrite later
    net.sgen['q_price'] = 0

    # Define the RL problem
    # See all load power values and sgen prices...
    obs_keys = [('load', 'p_mw', net['load'].index),
                ('load', 'q_mvar', net['load'].index),
                ('sgen', 'p_price', net['sgen'].index),
                ('sgen', 'q_price', net['sgen'].index)]
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
        q_costs = (net.sgen['q_price'] * net.sgen['q_mvar']**2).sum()
        p_costs = (net.sgen['p_price'] * net.sgen['p_mw']).sum()
        return -(q_costs + p_costs)

    pp.runpp(net)

    # Create the environment
    env = opf_env.OpfEnv(net, objective, obs_keys,
                         obs_space, act_keys, act_space)

    return env


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
