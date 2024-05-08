
from mlopf.envs.voltage_control import VoltageControl


class QMarket(VoltageControl):
    """
    Reactive power market environment (special case of VoltageControl): 
    The grid operator procures reactive power from generators to minimize 
    losses within its system. 

    Actuators: Reactive power of all gens

    Sensors: active+reactive power of all loads; active power of all gens;
        reactive prices of all gens

    Objective: minimize reactive power costs + minimize loss costs

    Constraints: Voltage band, line/trafo load, min/max reactive power,
        constrained reactive power flow over slack bus

    """

    def __init__(self, simbench_network_name='1-LV-urban6--0-sw',
                 gen_scaling=2.0, load_scaling=1.5, seed=None, min_obs=False,
                 cos_phi=0.9, max_q_exchange=0.01, market_based=True,
                 *args, **kwargs):

        super().__init__(simbench_network_name=simbench_network_name,
                         load_scaling=load_scaling, 
                         gen_scaling=gen_scaling,
                         cos_phi=cos_phi,
                         max_q_exchange=max_q_exchange, 
                         market_based=market_based,
                         seed=seed,
                         *args, **kwargs)

if __name__ == '__main__':
    env = QMarket()
    print('Reactive power market environment created')
    print('Observation space:', env.observation_space.shape)
    print('Action space:', env.action_space.shape)

    env = VoltageControl()
    print('VoltageControl environment created')
    print('Observation space:', env.observation_space.shape)
    print('Action space:', env.action_space.shape)