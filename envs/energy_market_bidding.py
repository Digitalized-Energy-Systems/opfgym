""" Reinforcement Learning environments to train multiple agents to bid on a
energy market environment (i.e. an economic dispatch). """

import gym
import numpy as np
import pandapower as pp

from .thesis_envs import EcoDispatchEnv


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
                 overload_penalty=1, penalty_factor=10, learn_bids=True,
                 reward_scaling=0.1, in_agent=False, *args, **kwargs):

        assert market_rules in ('pab', 'uniform')
        self.market_rules = market_rules
        self.in_agent = in_agent  # Compute part of reward within the RL algo
        self.penalty_factor = penalty_factor
        self.learn_bids = learn_bids
        self.reward_scaling = reward_scaling
        super().__init__(simbench_network_name, 0, n_agents,
                         gen_scaling, load_scaling, u_penalty, overload_penalty,
                         *args, **kwargs)

        self.internal_costs = 20  # Arbitrary values currently: 2 ct/kwh
        # TODO: Add marginal costs for the power plants (different for each!)
        self.max_power = np.array(self.net.sgen.max_p_mw)

        self.n_agents = len(self.sgen_idxs)
        n_rewards = len(self.sgen_idxs) + 1
        if self.in_agent:
            # TODO: Maybe move this adjustment to RL algo instead
            self.reward_space = gym.spaces.Box(
                low=-np.ones(1) * np.inf, high=np.ones(1) * np.inf)
        else:
            self.reward_space = gym.spaces.Box(
                low=-np.ones(n_rewards) * np.inf, high=np.ones(n_rewards) * np.inf)

        # Cost function to set penalty in OPF
        self.net.ext_grid['min_p_mw'] = -100
        self.net.ext_grid['max_p_mw'] = 100
        pp.create_pwl_cost(self.net, element=0, et='ext_grid',
                           points=[[-100, 0, 0],
                                   [0, 100, self.penalty_factor]],
                           power_type='p')
        # self.net.poly_cost = self.net.poly_cost.drop(0)
        # TODO: Maybe remove in base env? or update obs space (this is a potential error)

    def _set_action_space(self):
        """ Each power plant can be set in range from 0-100% power
        (minimal power higher than zero not considered here) """
        if self.in_agent:
            return super()._set_action_space()

        if self.market_rules == 'uniform':
            # Same as base environment, but market price as additional action
            low = np.zeros(
                len(self.act_keys[0][2]) + len(self.act_keys[1][2]) + 1)
            high = np.ones(
                len(self.act_keys[0][2]) + len(self.act_keys[1][2]) + 1)
        elif self.market_rules == 'lmp':
            raise NotImplementedError
        elif self.market_rules == 'pab':
            # Same as base environment: Only the setpoints
            # TODO: Maybe add bidding as actuator (instead of random sampling)
            if not self.learn_bids:
                return super()._set_action_space()
            else:
                low = np.zeros(
                    len(self.act_keys[0][2]) + len(self.act_keys[1][2]) + len(self.sgen_idxs))
                high = np.ones(
                    len(self.act_keys[0][2]) + len(self.act_keys[1][2]) + len(self.sgen_idxs))

        self.action_space = gym.spaces.Box(low, high)

    def step(self, action):
        # TODO: Overwrite these bids when they are learned within the algo!
        # self.net.poly_cost.cp1_eur_per_mw[self.net.poly_cost.et ==
        #                                   'ext_grid'] = 0
        self.bids = np.array(
            self.net.poly_cost.cp1_eur_per_mw[self.net.poly_cost.et == 'sgen'])
        if self.market_rules == 'uniform':
            self.market_price = action[-1] * self.max_price
            # Ignore setpoints from units that bid higher than market price
            action[:-1][self.bids > self.market_price] = 0.0
            self.setpoints = action[:-1]
            assert len(self.setpoints) == len(self.net.sgen)
        elif self.market_rules == 'pab':
            self.market_price = None
            if self.learn_bids:
                self.bids = action[-self.n_agents:] * self.max_price
                self.setpoints = action[:-self.n_agents]
                self.net.poly_cost.cp1_eur_per_mw[self.net.poly_cost.et ==
                                                  'sgen'] = self.bids * self.reward_scaling
            else:
                self.setpoints = action

        # print('env bids: ', list(self.bids))
        obs, reward, done, info = super().step(action=action)

        # TODO: What is the meaning of this?
        reward -= sum(info['penalty'])
        if self.vector_reward is True:
            reward = np.append(reward, np.array(info['penalty']))
        else:
            reward = np.append(reward, sum(info['penalty']))
        return obs, reward, done, info

    def _calc_reward(self, net):
        """ Create a reward vector (!) that consists of market profit for each
        agent """
        # TODO: Currently no objective function, except for cost min, but only constraint satisfaction
        if self.market_rules == 'uniform':
            return -self.market_price * np.array(self.net.res_sgen.p_mw)
        elif self.market_rules == 'lmp':
            raise NotImplementedError
        elif self.market_rules == 'pab':
            # Ignore "market price" completely here
            # Why setpoints? the actual power values make more sense
            return -self.bids * np.array(self.net.res_sgen.p_mw) * self.reward_scaling

    def _calc_penalty(self):
        penalty = super()._calc_penalty()
        # Do not allow to procure active power from superordinate system
        ext_grid_penalty = (sum(self.net.res_ext_grid.p_mw)
                            ) * self.penalty_factor  # 15
        # if ext_grid_penalty < -0.5:
        #     print('ext grid penalty: ', ext_grid_penalty)

        if sum(self.net.res_ext_grid.p_mw) < 0:
            # No negative penalties allowed
            ext_grid_penalty = 0

        penalty.append(-ext_grid_penalty)
        return penalty
