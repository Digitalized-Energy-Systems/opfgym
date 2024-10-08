import numpy as np
import simbench as sb


def build_simbench_net(simbench_network_name, gen_scaling=1.0, load_scaling=1.0,
                       storage_scaling=1.0, voltage_band=0.05, max_loading=80, 
                       *args, **kwargs):
    """ Init and return a simbench power network with standard configuration.
    """

    net = sb.get_simbench_net(simbench_network_name)

    set_unit_scaling(net, gen_scaling, load_scaling, storage_scaling)
    set_system_constraints(net, voltage_band, max_loading)

    assert not sb.profiles_are_missing(net)
    profiles = sb.get_absolute_values(
        net, profiles_instead_of_study_cases=True)

    repair_simbench_profiles(net, profiles)
    set_constraints_from_profiles(net, profiles)

    return net, profiles


def set_unit_scaling(net, gen_scaling=1.0, load_scaling=1.0, 
                     storage_scaling=1.0):
    net.sgen['scaling'] = gen_scaling
    net.gen['scaling'] = gen_scaling
    net.load['scaling'] = load_scaling
    net.storage['scaling'] = storage_scaling


def set_system_constraints(net, voltage_band=None, max_loading=None):
    # Define the voltage band of plus/minus `voltage_band`
    if voltage_band:
        net.bus['max_vm_pu'] = 1 + voltage_band
        net.bus['min_vm_pu'] = 1 - voltage_band
    # Set maximum loading of lines and transformers
    if max_loading:
        net.line['max_loading_percent'] = max_loading
        net.trafo['max_loading_percent'] = max_loading


def repair_simbench_profiles(net, profiles):
    """ The simbench data sometimes contains faulty data that needs to be
    repaired/thrown out. """

    # TODO: Bad style: this function does two things: repair profiles and set some constraints

    # Fix strange error in simbench: Sometimes negative active power values
    profiles[('sgen', 'p_mw')][profiles[('sgen', 'p_mw')] < 0.0] = 0.0

    # Another strange error: Sometimes min and max power are both zero
    # Remove these units from profile and pp net!
    for type_act in profiles.keys():
        net_df = net[type_act[0]]

        is_equal = profiles[type_act].max(
            axis=0) == profiles[type_act].min(axis=0)
        net_df.drop(net_df[is_equal].index, inplace=True)

        df = profiles[type_act]
        df.drop(columns=df.columns[is_equal], inplace=True)


def set_constraints_from_profiles(net, profiles):
    """ Set data boundaries from profiles as constraints for the OPF and RL
    problem definition. """
    for type_act in profiles.keys():
        unit_type, column = type_act
        net_df = net[unit_type]

        if unit_type == 'storage':
            max_power = np.maximum(profiles[type_act].max(axis=0).abs(),
                                   profiles[type_act].min(axis=0).abs())
            net_df[f'max_max_{column}'] = max_power * net_df.scaling
            net_df[f'min_min_{column}'] = -max_power * net_df.scaling
        else:
            net_df[f'max_max_{column}'] = profiles[type_act].max(axis=0) * net_df.scaling
            net_df[f'min_min_{column}'] = profiles[type_act].min(axis=0) * net_df.scaling
        # Compute mean and standard dev. Sometimes required for data sampling.
        net_df[f'mean_{column}'] = profiles[type_act].mean(axis=0)
        net_df[f'std_dev_{column}'] = profiles[type_act].std(axis=0)

    # Add estimation of min/max data for external grids
    load_gen_diff = profiles[('load', 'p_mw')].sum(
        axis=1) - profiles[('sgen', 'p_mw')].sum(axis=1)
    net.ext_grid['max_max_p_mw'] = load_gen_diff.max()
    net.ext_grid['min_min_p_mw'] = load_gen_diff.min()
    net.ext_grid['mean_p_mw'] = load_gen_diff.mean()
    # Generators should normally not increase q imbalances further
    # -> Only look at load reactive power!
    load_q_mvar = profiles[('load', 'q_mvar')].sum(axis=1)
    net.ext_grid['max_max_q_mvar'] = load_q_mvar.max()
    net.ext_grid['min_min_q_mvar'] = load_q_mvar.min()
    net.ext_grid['mean_q_mvar'] = load_q_mvar.mean()
