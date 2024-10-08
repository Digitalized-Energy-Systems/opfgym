
from opfgym.envs.voltage_control import VoltageControl


class QMarket(VoltageControl):
    """
    Reactive power market environment (special case of VoltageControl):
    The grid operator procures reactive power from generators to minimize 
    losses within its system.

    Actuators: Reactive power of the bigger generators in the system.

    Sensors: Active+reactive power of all loads; active power of all generators
        and storages; reactive power prices of the controllable generators.

    Objective: minimize reactive power costs + minimize loss costs

    Constraints: Voltage band, line/trafo load, min/max reactive power,
        constrained reactive power flow over slack bus.
    """

    def __init__(self, simbench_network_name='1-MV-rural--0-sw',
                 gen_scaling=1.0, load_scaling=1.3, seed=None, 
                 min_sgen_power=0.2, cos_phi=0.95, max_q_exchange=0.2, 
                 market_based=True, 
                 *args, **kwargs):

        super().__init__(simbench_network_name=simbench_network_name,
                         load_scaling=load_scaling, 
                         gen_scaling=gen_scaling,
                         cos_phi=cos_phi,
                         max_q_exchange=max_q_exchange, 
                         market_based=market_based,
                         min_sgen_power=min_sgen_power,
                         seed=seed,
                         *args, **kwargs)


if __name__ == '__main__':
    env = QMarket()
    print('Reactive power market environment created')
    print('Number of buses: ', len(env.net.bus))
    print('Observation space:', env.observation_space.shape)
    print('Action space:', env.action_space.shape, f'(Generators: {sum(env.net.sgen.controllable)}, Storage: {sum(env.net.storage.controllable)})')
