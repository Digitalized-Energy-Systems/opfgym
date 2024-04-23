""" Sanity check for newly created RL-OPF environments to verify that the 
environment and OPF definition make sense and aligned with each other (that 
reward maximization actually solves the OPF). """

import numpy as np


def env_sanity_check(env):
    env.reset()
    check_action_space(env.net, env.act_keys)
    # Add more checks here if needed
    return True


def check_action_space(net, act_keys: dict):
    """ Check if the action space of the environment is aligned with the number 
    and type of controllable units in the OPF definition. """

    # Assert that all RL actuators are controllable by the pandapower OPF
    for unit_type, column, idxs in act_keys:
        df = net[unit_type]

        assert df.controllable[idxs].all(), f'Not all RL actuators can be used by the pandapower OPF! -> The OPF is not solvable with the pandapower OPF!'
        assert df.in_service[idxs].all(), f'Not all RL actuators are in service! -> The OPF is not solvable with the pandapower OPF!'
        # Assert that constraints are set as well
        assert (df[f'min_{column}'] <= df[f'max_{column}']).all(), 'The min and max values of the OPF constraints are not consistent!'

    # Assert that all OPF controllable units are used by the RL environment
    # TODO: Do ext grids need to be tested as well?
    # TODO: How to differentiate between active and reactive power here? This framework does while pp does not differentiate.
    for unit_type_is in ('load', 'sgen', 'gen', 'storage'):
        df = net[unit_type_is]
        if len(df) == 0:
            continue

        controllable_idxs = set(df[df['controllable']].index)

        if len(controllable_idxs) == 0:
            continue

        # The controllable units should all exist in the act keys
        active_power_idxs_set = set()
        reactive_power_idxs_set = set()
        for unit_type_should, column, idxs in act_keys:
            if unit_type_is == unit_type_should:
                # These indices are controllable by the RL algorithm as well
                if column == 'p_mw':
                    active_power_idxs_set.update(set(idxs))
                elif column == 'q_mvar':
                    reactive_power_idxs_set.update(set(idxs))
                else:
                    raise NotImplementedError(f'Unknown column {column} for unit type {unit_type_should}!')

        total_idxs_set = active_power_idxs_set.union(reactive_power_idxs_set)
        # All controllable units should be used by the RL environment
        assert controllable_idxs == total_idxs_set, f'Not all controllable units of the OPF are used by the RL environment! -> The OPF is not solvable with RL!'
        
        # If only active power is usable, reactive power should be constrained tightly
        only_active_power = np.array(active_power_idxs_set - reactive_power_idxs_set)
        assert ((df.max_q_mvar.loc[only_active_power] - df.min_q_mvar.loc[only_active_power]) < 1e-6).all()

        # And vice versa 
        only_reactive_power = np.array(reactive_power_idxs_set - active_power_idxs_set)
        assert ((df.max_p_mw.loc[only_reactive_power] - df.min_p_mw.loc[only_reactive_power]) < 1e-6).all()


if __name__ == '__main__':
    from mlopf.envs import MaxRenewable, EcoDispatch, QMarket, VoltageControl, LoadShedding
    for env_class in (MaxRenewable, EcoDispatch, QMarket, VoltageControl, LoadShedding):
        print('Test environment:', env_class.__name__)
        env = env_class()
        env_sanity_check(env)
