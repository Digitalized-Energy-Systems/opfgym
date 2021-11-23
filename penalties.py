import numpy as np

# TODO: Lots of code repetition here -> one function that gets called by all!


def voltage_violation(net, penalty_factor):
    """ Linear penalty for voltage violations of the upper or lower voltage
    boundary (both treated equally). """
    # TODO: implement this whole stuff only once: violation(unit_type=bus, column=vm_pu, constrain_column=etc)
    voltages = net.res_bus.vm_pu.to_numpy()
    max_voltages = net.bus["max_vm_pu"].to_numpy()
    min_voltages = net.bus["min_vm_pu"].to_numpy()

    upper_mask = voltages > max_voltages
    lower_mask = voltages < min_voltages

    upper_violations = (voltages - max_voltages)[upper_mask].sum()
    lower_violations = (min_voltages - voltages)[lower_mask].sum()
    # TODO: use vector instead
    penalty = (upper_violations + lower_violations) * penalty_factor
    if upper_violations > 0:
        print('overvoltage: ', upper_violations * penalty_factor)
        # print(voltages)
    if lower_violations > 0:
        print('undervoltage: ', lower_violations * penalty_factor)
        # print(voltages)

    return penalty


def line_trafo_overload(net, penalty_factor, unit_type: str):
    loads = net[f'res_{unit_type}'].loading_percent.to_numpy()
    max_loads = net[unit_type].max_loading_percent.to_numpy()

    mask = loads > max_loads
    violations = (loads - max_loads)[mask].sum()
    if violations > 0:
        print(f'{unit_type} overload: ', violations * penalty_factor)
    return violations * penalty_factor


def ext_grid_overpower(net, penalty_factor, column='q_mvar'):
    """ Linear penalty for violations of max/min active/reactive power from
    external grids. """
    power = net.res_ext_grid[column].to_numpy()
    max_power = net.ext_grid[f'max_{column}'].to_numpy()
    min_power = net.ext_grid[f'min_{column}'].to_numpy()

    upper_mask = power > max_power
    lower_mask = power < min_power

    upper_violations = (power - max_power)[upper_mask].sum()
    lower_violations = (min_power - power)[lower_mask].sum()

    penalty = (upper_violations + lower_violations) * penalty_factor
    if penalty > 0:
        print(f'External grid {column} violated: ', penalty)
    return penalty


def apparent_overpower(net, penalty_factor, autocorrect=True):
    power = (net.sgen.p_mw.to_numpy() ** 2 +
             net.sgen.q_mvar.to_numpy() ** 2)**0.5
    max_power = net.sgen.max_s_mva.to_numpy()

    # TODO: 'res_sgen`instead?!

    mask = power > max_power
    violations = (power - max_power)[mask].sum()
    # if violations > 0.00000:
    #     print('apparent power over max: ', violations * penalty_factor)

    if autocorrect:
        correct_apparent_overpower(net)

    return violations * penalty_factor


def correct_apparent_overpower(net):
    """ Apparent power is not automatically bounded by the agent. Invalid
    actions need to be ignored, if necessary. Assumption: Always reduce
    reactive power, if apparent power is too high. """
    s_mva2 = net.sgen.max_s_mva.to_numpy() ** 2
    p_mw2 = net.sgen.p_mw.to_numpy() ** 2
    q_mvar_max = (s_mva2 - p_mw2)**0.5
    new_values = np.minimum(net.sgen['q_mvar'].abs(), q_mvar_max)
    net.sgen['q_mvar'] = np.sign(net.sgen['q_mvar']) * new_values


def active_reactive_overpower(net, penalty_factor, column='p_mw',
                              autocorrect=True):
    power = net.res_sgen[column].to_numpy()
    max_power = net.sgen[f'max_{column}'].to_numpy()
    mask = power > max_power
    violations = (power - max_power)[mask].sum()
    # if violations > 0:
    #     print(f'{column} power over max: ', violations * penalty_factor)

    if autocorrect:
        correct_active_reactive_overpower(net, column)

    return violations * penalty_factor


def correct_active_reactive_overpower(net, column):
    """ Active power is not automatically bounded by the agent. Invalid
    actions need to be ignored, if necessary. """
    new_values = np.minimum(
        net.sgen[column].to_numpy(), net.sgen[f'max_{column}'].to_numpy())
    net.sgen[column] = new_values
