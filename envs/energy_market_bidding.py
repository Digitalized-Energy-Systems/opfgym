""" Reinforcement Learning environments to train multiple agents to bid on a
energy market environment (i.e. an economic dispatch). """

import gym
import numpy as np

from .thesis_envs import EcoDispatchEnv
from ..opf_env import get_obs_space


class OpfAndBiddingEcoDispatchEnv(EcoDispatchEnv):
    """ Special case: The grid operator learns optimal procurement of active
    energy (economic dispatch), while (multiple) market participants learn to
    bid on the market concurrently.

    TODO: Maybe this should not be a single-step env, because the agents can
    collect important information from the history of observations (eg voltages)
    TODO: Not really a general case. Maybe move to diss repo?!

    Actuators: TODO Not clearly defined yet

    Sensors: TODO Not clearly defined yet

    Objective: TODO Not clearly defined yet

    """

    def __init__(self, simbench_network_name='1-HV-urban--0-sw',
                 market_rules='uniform', n_agents=None,
                 load_scaling=1.5, gen_scaling=1.5, u_penalty=300,
                 overload_penalty=1):
        self.market_rules = market_rules
        super().__init__(simbench_network_name, 0, n_agents,
                         load_scaling, gen_scaling, u_penalty, overload_penalty)

        # TODO: Use observation mapping instead

        # Overwrite observation space
        # Handle last set of observations internally (the agents' bids)
        if self.market_rules == 'in_agent':
            self.obs_keys = self.obs_keys[0:-1]
            self.observation_space = get_obs_space(self.net, self.obs_keys)
        # TODO: Adjust observation space -> add bids

        self.internal_costs = 20  # Arbitrary values currently: 2 ct/kwh
        # TODO: Add marginal costs for the power plants (different for each!)

        n_rewards = len(self.sgen_idxs) + 1
        self.reward_space = gym.spaces.Box(
            low=-np.ones(n_rewards) * np.inf, high=np.ones(n_rewards) * np.inf)

    def _set_action_space(self):
        """ Each power plant can be set in range from 0-100% power
        (minimal power higher than zero not considered here) """
        if self.market_rules == 'uniform':
            # Same as base environment, but market price as additional action
            low = np.zeros(
                len(self.act_keys[0][2]) + len(self.act_keys[1][2]) + 1)
            high = np.ones(
                len(self.act_keys[0][2]) + len(self.act_keys[1][2]) + 1)
        elif self.market_rules == 'lmp':
            raise NotImplementedError
        elif self.market_rules == 'pab':
            raise NotImplementedError

        self.action_space = gym.spaces.Box(low, high)

    def step(self, action):
        # TODO: Overwrite these bids when the are learned within the algo!
        self.bids = self.net.poly_cost.cp1_eur_per_mw[self.net.poly_cost.et == 'sgen']
        if self.market_rules == 'uniform':

            self.market_price = action[-1]
            action[:-1][(self.bids / self.max_price) > self.market_price] = 0.0
            self.setpoints = action[:-1]
            assert len(self.setpoints) == len(self.net.sgen)

        obs, reward, done, info = super().step(action=action)

        if not self.market_rules == 'in_agent':
            reward += info['penalty']
            reward = np.append(reward, -info['penalty'])

        return obs, reward, done, info

    def _calc_reward(self, net):
        """ Create a reward vector (!) that consists of market profit for each
        agent """
        if self.market_rules == 'uniform':
            return -self.market_price * self.setpoints
        elif self.market_rules == 'lmp':
            raise NotImplementedError
        elif self.market_rules == 'pab':
            raise NotImplementedError
        elif self.market_rules == 'in_agent':
            # Everything is handled within the RL algo
            return 0

    def _calc_penalty(self):
        penalty = super()._calc_penalty()
        # Do not allow to procure active power from superordinate system
        # -> linear increasing penalty
        ext_grid_penalty = sum(self.net.res_ext_grid.p_mw) * 10
        if ext_grid_penalty > 1:
            print('ext grid penalty: ', ext_grid_penalty)

        if ext_grid_penalty < 0:
            # No negative penalties allowed
            return penalty

        penalty.append(-ext_grid_penalty)
        return penalty
