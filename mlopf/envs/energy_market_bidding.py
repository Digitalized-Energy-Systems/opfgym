""" Reinforcement Learning environments to train multiple agents to bid on a
energy market environment (i.e. an economic dispatch). """


import gym
import numpy as np
import pandapower as pp
import pettingzoo

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
                 market_rules='pab', n_agents=None,
                 load_scaling=3.0, gen_scaling=1.5, u_penalty=50,
                 overload_penalty=0.2, penalty_factor=600, learn_bids=True,
                 reward_scaling=0.0001, in_agent=False, uniform_gen_size=True,
                 other_bids='fixed', one_gen_per_agent=True,
                 rel_marginal_costs=0.1,
                 consider_marginal_costs=True, bid_as_reward=False,
                 remove_gen_idxs=None, *args, **kwargs):

        assert market_rules in ('pab', 'uniform')
        self.market_rules = market_rules
        self.in_agent = in_agent  # Compute part of reward within the RL algo
        self.penalty_factor = penalty_factor
        self.learn_bids = learn_bids
        self.reward_scaling = reward_scaling
        self.other_bids = other_bids
        self.uniform_gen_size = uniform_gen_size
        self.one_gen_per_agent = one_gen_per_agent
        self.rel_marginal_costs = rel_marginal_costs
        self.consider_marginal_costs = consider_marginal_costs
        self.bid_as_reward = bid_as_reward
        self.remove_gen_idxs = remove_gen_idxs  # default: Remove randomly
        self._seed = kwargs['seed']

        self.n_agents = n_agents
        self.agent_idxs = np.arange(n_agents)

        super().__init__(simbench_network_name, 0, n_agents,
                         gen_scaling, load_scaling, *args, **kwargs)
        # Overwrite action space
        self._set_action_space(self._seed)

        self.internal_costs = 20  # Arbitrary value currently: 2 ct/kwh
        # TODO: Add marginal costs for the power plants? (different for each!)
        self.max_power = np.array(self.net.sgen.max_p_mw)
        self.n_gens = len(self.net.sgen.index)

        if self.vector_reward:
            n_rewards = self.n_gens + 5 + 1
        else:
            n_rewards = self.n_gens + 1
        if self.in_agent:
            # TODO: Maybe move this adjustment to RL algo instead
            self.reward_space = gym.spaces.Box(
                low=-np.ones(1) * np.inf,
                high=np.ones(1) * np.inf,
                seed=self._seed)
        else:
            self.reward_space = gym.spaces.Box(
                low=-np.ones(n_rewards) * np.inf,
                high=np.ones(n_rewards) * np.inf,
                seed=self._seed)

    def _build_net(self, *args, **kwargs):
        net = super()._build_net(*args, **kwargs)

        net.ext_grid['controllable'] = True
        # Cost function to set penalty in OPF
        net.ext_grid['min_p_mw'] = -10000
        net.ext_grid['max_p_mw'] = 10000
        pp.create_pwl_cost(net, element=0, et='ext_grid',
                           points=[[-10000, 0, 0],
                                   [0, 10000, self.penalty_factor]],
                           power_type='p')
        # Remove poly cost instead
        net.poly_cost = net.poly_cost.drop(0)
        # TODO: Maybe remove in base env? or update obs space (this is a potential error)

        if self.uniform_gen_size:
            # Set all generators to the same size
            net.sgen.max_p_mw = net.sgen.max_p_mw.mean()
            net.sgen.max_max_p_mw = net.sgen.max_max_p_mw.mean()

        if self.one_gen_per_agent:
            old_n_gens = len(net.sgen.index)
            # Remove random generators so that there is one generator per agent
            if self.remove_gen_idxs is None:
                self.remove_gen_idxs = np.random.choice(
                    net.sgen.index, len(net.sgen.index) - self.n_agents, replace=False)

            print('Remove generators: ', self.remove_gen_idxs)
            net.sgen = net.sgen.drop(self.remove_gen_idxs)
            net.poly_cost = net.poly_cost.drop(
                net.poly_cost.index[net.poly_cost.element.isin(self.remove_gen_idxs)])
            if self.uniform_gen_size:
                # Increase power of remaining gens so that it stays constant
                net.sgen.max_p_mw = net.sgen.max_p_mw * \
                    old_n_gens / self.n_agents
                net.sgen.max_max_p_mw = net.sgen.max_max_p_mw.mean() * old_n_gens / \
                    self.n_agents

        return net

    def _set_action_space(self, seed):
        """ Each power plant can be set in range from 0-100% power
        (minimal power higher than zero not considered here) """
        if self.in_agent:
            low = np.zeros(len(self.act_keys[0][2]) + len(self.act_keys[1][2]))
            high = np.ones(len(self.act_keys[0][2]) + len(self.act_keys[1][2]))

        elif self.one_gen_per_agent and self.market_rules == 'pab':
            low = np.zeros(self.n_agents * 2)
            high = np.ones(self.n_agents * 2)

        elif self.market_rules == 'uniform':
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
                low = np.zeros(
                    len(self.act_keys[0][2]) + len(self.act_keys[1][2]))
                high = np.ones(
                    len(self.act_keys[0][2]) + len(self.act_keys[1][2]))
            else:
                low = np.zeros(
                    len(self.act_keys[0][2]) + len(self.act_keys[1][2]) + self.n_agents)
                high = np.ones(
                    len(self.act_keys[0][2]) + len(self.act_keys[1][2]) + self.n_agents)

        self.action_space = gym.spaces.Box(low, high, seed=seed)

    def step(self, action, test=False):
        # TODO: Overwrite bids when learned within the algo! (otherwise random)
        if self.other_bids == 'fixed':
            self.net.poly_cost.cp1_eur_per_mw[self.net.poly_cost.et ==
                                              'sgen'] = self.max_price / 4 * self.reward_scaling
        elif self.other_bids == 'noisy_fixed':
            if not test:
                self.net.poly_cost.cp1_eur_per_mw[
                    self.net.poly_cost.et == 'sgen'] = (1 / 4 + np.random.randn(self.n_gens) * 0.1) * \
                    self.max_price * self.reward_scaling
            else:
                self.net.poly_cost.cp1_eur_per_mw[
                    self.net.poly_cost.et == 'sgen'] = 1 / 4 * self.max_price * self.reward_scaling
        elif self.other_bids == 'average':
            self.net.poly_cost.cp1_eur_per_mw[self.net.poly_cost.et ==
                                              'sgen'] = np.mean(action[-self.n_agents:]) * self.max_price * self.reward_scaling

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
                self.bids[self.agent_idxs] = action[-self.n_agents:] * \
                    self.max_price * self.reward_scaling
                self.setpoints = action[:len(self.net.sgen.index)]
                self.net.poly_cost.cp1_eur_per_mw[
                    self.net.poly_cost.et == 'sgen'] = self.bids / self.reward_scaling
            else:
                self.setpoints = action

        # print('env bids: ', list(self.bids))
        obs, reward, done, info = super().step(action=action)

        if self.vector_reward is not True:
            reward -= sum(info['penalties'])
            reward = np.append(reward, sum(info['penalties']))

        return obs, reward, done, info

    # TODO: Currently buggy
    # def _calc_objective(self, net):
    #     """ Create a reward vector (!) that consists of market profit for each
    #     agent """
    #     # TODO: Currently no objective function, except for cost min, but only constraint satisfaction
    #     if self.market_rules == 'uniform':
    #         return -self.market_price * np.array(self.net.res_sgen.p_mw)
    #     elif self.market_rules == 'lmp':
    #         raise NotImplementedError
    #     elif self.market_rules == 'pab':
    #         # Ignore "market price" completely here
    #         rewards = -self.bids * np.array(self.net.res_sgen.p_mw)
    #         if self.consider_marginal_costs:
    #             rewards += self.rel_marginal_costs * self.reward_scaling * \
    #                 self.max_price * np.array(self.net.res_sgen.p_mw)

    #         # To prevent zero gradient -> bid as negative reward if bid too high
    #         # The OPF often fails to set the setpoints to exactly zero
    #         # TODO: Maybe make this optional
    #         rel_setpoints = np.array(
    #             self.net.res_sgen.p_mw / self.net.sgen.max_p_mw)
    #         if self.bid_as_reward is True:
    #             rewards[rel_setpoints < 0.05] += self.bids[rel_setpoints < 0.05]
    #         else:
    #             rewards[rel_setpoints < 0.001] = 0.0

    #         return rewards

    def _calc_penalty(self):
        penalties, valids = super()._calc_penalty()
        # Do not allow to procure active power from superordinate system
        if sum(self.net.res_ext_grid.p_mw) < 0:
            # No negative penalties allowed
            ext_grid_penalty = 0.0
        else:
            ext_grid_penalty = (sum(self.net.res_ext_grid.p_mw)
                                ) * self.penalty_factor * self.reward_scaling

        # if ext_grid_penalty > 1.0:
        #     print('ext grid penalty: ', ext_grid_penalty)

        penalties.append(-ext_grid_penalty)
        # Soft constraint: Always valid!
        valids.append(True)

        return penalties, valids


class OpfAndBiddingEcoDispatchEnvBaseMarl(OpfAndBiddingEcoDispatchEnv):
    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(seed=seed, *args, **kwargs)
        # Each agent has one reward, six observations, and one action
        self.agent_reward_mapping = np.array(range(self.n_agents))
        # Assumption: Agents only know current time, but nothing about system
        self.agent_obs_mapping = [np.arange(6) for _ in range(self.n_agents)]
        # One action per agent, ie its respective bid
        self.agent_action_mapping = np.array(range(self.n_agents))

        # Overwrite action space with bid-actuators
        self.act_keys = [
            ('poly_cost', 'cp1_eur_per_mw', self.net.poly_cost.index)]
        low = np.zeros(len(self.net.sgen.index))
        high = np.ones(len(self.net.sgen.index))
        self.action_space = gym.spaces.Box(
            low, high, seed=self._seed)
        # Define what 100% as action means -> max price!
        self.net.poly_cost['max_max_cp1_eur_per_mw'] = (
            self.net.poly_cost.max_cp1_eur_per_mw)

        # No powerflow calculation is required after reset (saves computation)
        self.res_for_obs = False

    def _calc_penalty(self):
        # The agents do not care about grid constraints -> no penalty!
        return []

    def _run_pf(self):
        """ Run not only a powerflow but an optimal power flow as proxy for
        the grid operator's behavior. """
        return self._optimal_power_flow()


class BiddingEcoDispatchEnv(pettingzoo.ParallelEnv):
    """ Multi-agent interface to single agent environment
    OpfAndBiddingEcoDispatchEnvBaseMarl. Converts reward/obs/action arrays to
    dicts.

    """

    metadata = {"name": "BiddingEcoDispatchEnv"}

    def __init__(self, *args, **kwargs):
        # This env is essentially a wrapper around a gym environment
        self.internal_env = OpfAndBiddingEcoDispatchEnvBaseMarl(
            *args, **kwargs)

        # Every generator is one agent that participates in the market
        self.possible_agents = [
            f'gen_{idx}' for idx in self.internal_env.net.sgen.index]
        self.agents = self.possible_agents

        self.observation_spaces = {
            a_id: gym.spaces.Box(
                low=self.internal_env.observation_space.low[self.internal_env.agent_obs_mapping[idx]],
                high=self.internal_env.observation_space.high[self.internal_env.agent_obs_mapping[idx]],
                seed=self.internal_env._seed)
            for idx, a_id in enumerate(self.agents)}

        # Each agent has one actuator: its bidding price on the market
        self.action_spaces = {
            a_id: gym.spaces.Box(
                low=-np.zeros(1), high=np.ones(1), seed=self.internal_env._seed)
            for idx, a_id in enumerate(self.agents)}

        self.state_space = self.internal_env.observation_space

    def reset(self, step=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        # self.state = {agent: NONE for agent in self.agents}
        # self.observations = {agent: NONE for agent in self.agents}
        self.num_moves = 0

        obss_array = self.internal_env.reset(step)

        return self._obs_to_dict(obss_array)

    def step(self, actions: dict):
        # TODO: Use action mapping here!
        actions_array = np.concatenate(
            [actions[a_id] for a_id in self.possible_agents])
        obss_array, rewards_array, done, info = self.internal_env.step(
            actions_array)

        # Minus sign because normally these are rewards for grid operator
        rewards = {a_id: -rewards_array[self.internal_env.agent_reward_mapping[idx]]
                   for idx, a_id in enumerate(self.possible_agents)}

        obss = self._obs_to_dict(obss_array)

        dones = {a_id: done for a_id in self.possible_agents}
        infos = {a_id: info for a_id in self.possible_agents}

        return obss, rewards, dones, infos

    def _obs_to_dict(self, obss_array):
        return {a_id: obss_array[self.internal_env.agent_obs_mapping[idx]]
                for idx, a_id in enumerate(self.possible_agents)}

    def state(self):
        return self.internal_env._get_obs()

    def render(self, mode):
        self.internal_env(mode)

    def close(self):
        pass

    def action_space(self, agent):
        return self.action_spaces[agent]
