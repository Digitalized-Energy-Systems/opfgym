""" Reinforcement Learning environment to train multiple agents to bid on a
reactive power market environment. """

import gym
import numpy as np

from ..thesis_envs import QMarketEnv
from ..opf_env import get_obs_space
from ..objectives import min_p_loss

import random


class BiddingQMarketEnv(QMarketEnv):
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
        profits = np.array((net.poly_cost['cq2_eur_per_mvar2'].loc[net.sgen.index]
                            * net.res_sgen['q_mvar']**2))

        if random.random() < 0.001:
            print('actions: ', list(net.poly_cost['cq2_eur_per_mvar2']))

        return profits

    def _calc_penalty(self):
        # The agents do not care about grid constraints -> no penalty!
        return 0

    def _run_pf(self):
        """ Run not only a powerflow but an optimal power flow as proxy for
        the grid operator's behavior. """
        self._optimal_power_flow()


# TODO: This is still work in progress
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
        # Handle last set of observations internally (the agents' bids)
        # TODO: Use observation mapping instead
        self.obs_keys = self.obs_keys[0:-1]
        self.observation_space = get_obs_space(self.net, self.obs_keys)

    def _calc_reward(self, net):
        """ Consider quadratic reactive power costs on the market and linear
        active costs for losses in the system. """
        # The agents handle their trading internally here
        q_costs = 0

        # Grid operator also wants to minimize network active power losses
        loss_costs = min_p_loss(net) * self.loss_costs

        return -q_costs - loss_costs
