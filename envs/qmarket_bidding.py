""" Reinforcement Learning environment to train multiple agents to bid on a
reactive power market environment. """

import gym
import numpy as np

from ..thesis_envs import QMarketEnv


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
        self.agent_observation_mapping = [
            np.array([idx]) for idx in self.net.sgen.index]
        self.agent_action_mapping = self.agent_observation_mapping

        # Overwrite action space with bid-actuators
        self.act_keys = [
            ('poly_cost', 'cq2_eur_per_mvar2', self.net['sgen'].index)]
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

        return profits

    def _calc_penalty(self):
        # The agents do not care about grid constraints -> no penalty!
        return 0

    def _run_pf(self):
        """ Run not only a powerflow but an optimal power flow as proxy for
        the grid operator's behavior. """
        self._optimal_power_flow()
