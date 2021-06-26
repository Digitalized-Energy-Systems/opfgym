import gym
import numpy as np
import pandapower.networks as pn
import pandapower as pp

from . import opf_env
from .objectives import min_p_loss

""" TODO: Create more examples and use as benchmark later
- Prices as observations (eg eco dispatch/q-market)
- Multi-step problems (storage systems as actuators? Ramps? Controllable loads?)
- Not fully observable (some measurements missing) or measurements that get
influenced by actions (eg. voltages)
- Other actuators: controllable loads, storage systems
"""


def example1():
    """ Some standard non-economic single-step OPF:

    Actuators: active and reactive power of all gens
    Sensors: active and reactive power of all loads
    Objective: minimize losses
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
    net.load['max_p_mw'] = net.load['p_mw'] * 0.5
    net.load['min_p_mw'] = net.load['p_mw'] * 0.05
    net.load['max_q_mvar'] = net.load['max_p_mw'] * 0.3
    net.load['min_q_mvar'] = net.load['min_p_mw'] * 0.3
    # ...for actions
    net.sgen['max_p_mw'] = net.sgen['p_mw']
    net.sgen['min_p_mw'] = np.zeros(len(net.sgen.index))
    net.sgen['max_s_mva'] = net.sgen['max_p_mw'] / 0.95  # = cos phi
    net.sgen['max_q_mvar'] = net.sgen['max_s_mva']
    net.sgen['min_q_mvar'] = -net.sgen['max_s_mva']

    # Define the RL problem
    # See all loads and ...
    obs_keys = [('load', 'p_mw', net['load'].index),
                ('load', 'q_mvar', net['load'].index)]
    p_low = net.load['min_p_mw'].to_numpy()
    p_high = net.load['max_p_mw'].to_numpy()
    q_low = net.load['min_q_mvar'].to_numpy()
    q_high = net.load['max_q_mvar'].to_numpy()
    obs_space = gym.spaces.Box(
        np.append(p_low, q_low), np.append(p_high, q_high))

    # ... control all sgens (everything else assumed to be constant)
    act_keys = [('sgen', 'p_mw', net['sgen'].index),
                ('sgen', 'q_mvar', net['sgen'].index)]
    low = np.append(np.zeros(len(net['sgen'].index)),
                    -np.ones(len(net['sgen'].index)))
    high = np.ones(2 * len(net['sgen'].index))
    act_space = gym.spaces.Box(low, high)

    def objective(net):
        """ Formulate objective function as maximization problem """
        return -min_p_loss(net)

    pp.runpp(net)

    # Create the environment
    env = opf_env.OpfEnv(net, objective, obs_keys,
                         obs_space, act_keys, act_space)

    return env


if __name__ == '__main__':
    env = example1()
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
