""" Reinforcement Learning environment to train multiple agents to bid on a
reactive power market environment. """

import gym
import numpy as np
import pettingzoo

from .thesis_envs import QMarketEnv
from ..opf_env import get_obs_space
from ..objectives import min_p_loss

import random


class BiddingQMarketEnv(pettingzoo.ParallelEnv):
    """ Special case: Not the grid operator learns optimal procurement of
    reactive power, but (multiple) market participants learn to bid on the
    market.
    TODO: Maybe this should not be a single-step env, because the agents can
    collect important information from the history of observations (eg voltages)

    Actuators: Reactive power bid of each generator respectively

    Sensors: Local active power of each generator respectively

    Objective: maximize profit of each generator respectively

    """

    # TODO: Maybe consider slack as actual learning agent??????!!!!!!!!!!
    metadata = {"name": "BiddingQMarketEnv"}

    def __init__(self, simbench_network_name='1-LV-urban6--0-sw'):
        # This env is essentially a wrapper around a gym environment
        self.internal_env = BiddingQMarketEnvBase(simbench_network_name)

        # Every generator is one agent that participates in the market
        self.possible_agents = [
            f'gen_{idx}' for idx in self.internal_env.net.sgen.index]
        self.agents = self.possible_agents

        # Each agent only sees the active power feed-in of its own generator
        # TODO: Maybe add time and local voltage
        # TODO: Better use mapping here, in case of changes
        min_power = self.internal_env.observation_space.low
        max_power = self.internal_env.observation_space.high
        self.observation_spaces = {
            a_id: gym.spaces.Box(
                low=min_power[idx:idx + 1], high=max_power[idx:idx + 1])
            for idx, a_id in enumerate(self.agents)}

        # Each agent has one actuator: its bidding price on the market
        self.action_spaces = {
            a_id: gym.spaces.Box(low=-np.zeros(1), high=np.ones(1))
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

        obss_array = self.internal_env.reset()

        return self._obs_to_dict(obss_array)

    def step(self, actions: dict):
        actions_array = np.concatenate(
            [actions[a_id] for a_id in self.possible_agents])
        obss_array, rewards_array, done, info = self.internal_env.step(
            actions_array)

        rewards = {a_id: rewards_array[idx]
                   for idx, a_id in enumerate(self.possible_agents)}

        obss = self._obs_to_dict(obss_array)

        if done:
            dones = {a_id: True for a_id in self.possible_agents}

        infos = {a_id: info for a_id in self.possible_agents}

        return obss, rewards, dones, infos

    def _obs_to_dict(self, obss_array):
        # TODO: better use the mapping here?!
        return {a_id: np.array(obss_array[idx]).reshape(1)
                for idx, a_id in enumerate(self.possible_agents)}

    def state(self):
        return self.internal_env._get_obs()

    def render(self, mode):
        self.internal_env(mode)

    def close(self):
        pass

    def action_space(self, agent):
        return self.action_spaces[agent]


# TODO: This is outdated -> use pettingzoo env instead for MARL
class BiddingQMarketEnvBase(QMarketEnv):
    """ Special case: Not the grid operator learns optimal procurement of
    reactive power, but (multiple) market participants learn to bid on the
    market.
    TODO: Maybe this should not be a single-step env, because the agents can
    collect important information from the history of observations (eg voltages)

    Actuators: Reactive power bid of each generator respectively

    Sensors: Local active power of each generator respectively

    Objective: maximize profit of each generator respectively

    """

    def __init__(self, simbench_network_name='1-LV-urban6--0-sw'):
        super().__init__(simbench_network_name='1-LV-urban6--0-sw')
        # Each agent has one reward, one observation, and one action
        self.agent_reward_mapping = np.array(self.net.sgen.index)
        # TODO: Add current time as observation!
        # Each agent only observes its own active power feed-in (see superclass)
        self.agent_observation_mapping = [
            np.array([idx]) for idx in self.net.sgen.index]
        self.agent_action_mapping = self.agent_observation_mapping

        # Overwrite action space with bid-actuators
        self.act_keys = [
            ('poly_cost', 'cq2_eur_per_mvar2', self.net.sgen.index)]
        low = np.zeros(len(self.net.sgen.index))
        high = np.ones(len(self.net.sgen.index))
        self.action_space = gym.spaces.Box(low, high)
        # Define what 100% as action means -> max price!
        self.net.poly_cost['max_max_cq2_eur_per_mvar2'] = (
            self.net.poly_cost.max_cq2_eur_per_mvar2)

        # No powerflow calculation is required after reset (saves computation)
        self.res_for_obs = False

    def _calc_reward(self, net):
        """ Consider quadratic reactive power profits on the market for each
        agent/generator. """
        self.internal_costs = 250  # 250 €/Mvar²h
        reactive_prices = net.poly_cost['cq2_eur_per_mvar2'].loc[net.sgen.index]
        profits = np.array(
            (reactive_prices - self.internal_costs) * net.res_sgen['q_mvar']**2)

        return profits

    def _calc_penalty(self):
        # The agents do not care about grid constraints -> no penalty!
        return []

    def _run_pf(self):
        """ Run not only a powerflow but an optimal power flow as proxy for
        the grid operator's behavior. """
        return self._optimal_power_flow()


class OpfAndBiddingQMarketEnv(QMarketEnv):
    """ Special case: The grid operator learns optimal procurement of
    reactive power, while (multiple) market participants learn to bid on the
    market concurrently.
    TODO: Maybe this should not be a single-step env, because the agents can
    collect important information from the history of observations (eg voltages)

    Actuators: TODO Not clearly defined yet

    Sensors: TODO Not clearly defined yet

    Objective: TODO Not clearly defined yet

    """

    def __init__(self, simbench_network_name='1-LV-urban6--0-sw'):
        super().__init__(simbench_network_name)

        # TODO: Use observation mapping instead
        # Overwrite observation space
        # Handle last set of observations internally (the agents' bids)
        self.obs_keys = self.obs_keys[0:-1]
        self.observation_space = get_obs_space(self.net, self.obs_keys)

        self.internal_costs = 250

    def _calc_reward(self, net):
        """ Consider quadratic reactive power costs on the market and linear
        active costs for losses in the system. """

        # The agents handle their trading internally here
        q_costs = 0
        if (self.net.poly_cost.et == 'ext_grid').any():
            mask = self.net.poly_cost.et == 'ext_grid'
            prices = self.net.poly_cost.cq2_eur_per_mvar2[mask].to_numpy()
            q_mvar = self.net.res_ext_grid.q_mvar.to_numpy()
            q_costs += sum(prices * q_mvar**2)

        # Grid operator also wants to minimize network active power losses
        loss_costs = min_p_loss(net) * self.loss_costs

        return -q_costs - loss_costs
