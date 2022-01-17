""" Reinforcement Learning environments to train multiple agents to bid on a
energy market environment (i.e. an economic dispatch). """


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

    def __init__(self, simbench_network_name='1-HV-urban--0-sw', n_agents=None):
        super().__init__(simbench_network_name, 0, n_agents)

        # TODO!!!: Allow to set number of participating power plants (agents) -> biggest power plants

        # TODO: Use observation mapping instead

        # Overwrite observation space
        # Handle last set of observations internally (the agents' bids)
        self.obs_keys = self.obs_keys[0:-1]
        self.observation_space = get_obs_space(self.net, self.obs_keys)

        self.internal_costs = 20  # Arbitrary values currently: 2 ct/kwh
        # TODO: Marginal costs for the power plants

    def _calc_reward(self, net):
        """ The only costs are market costs (handled internally) and penalties
        (handled in separate method). """

        return 0
