import numpy as np

# TODO: Lots of code repetition here -> one function that gets called by all!
# TODO: Does `autocorrect` require an additonal pf afterwards?


def calc_penalty(net, unit_type, column, constraint_column, maximal=True,
                 vectorize=False):
    values = net[unit_type][column].to_numpy()
    if 'res_' in unit_type:
        unit_type = unit_type[4:]
    constraints = net[unit_type][constraint_column].to_numpy()

    if maximal:
        violations = values - constraints
    else:
        violations = constraints - values
    violations[violations < 0] = 0.0

    if vectorize:
        return violations

    return sum(violations)


def voltage_violation(net, penalty_factor):
    """ Linear penalty for voltage violations of the upper or lower voltage
    boundary (both treated equally). """

    overvoltage = calc_penalty(net, 'res_bus', 'vm_pu', 'max_vm_pu')
    undervoltage = calc_penalty(net, 'res_bus', 'vm_pu', 'min_vm_pu', False)

    if overvoltage > 0:
        print('overvoltage: ', overvoltage * penalty_factor)
    if undervoltage > 0:
        print('undervoltage: ', undervoltage * penalty_factor)

    return penalty_factor * (overvoltage + undervoltage)


def line_trafo_overload(net, penalty_factor, unit_type: str, vectorize=False):
    violations = calc_penalty(net, f'res_{unit_type}', 'loading_percent',
                              'max_loading_percent', vectorize=vectorize)

    if violations > 0:
        print(f'{unit_type} overload: ', violations * penalty_factor)
    return violations * penalty_factor


def ext_grid_overpower(net, penalty_factor, column='q_mvar', vectorize=False):
    """ Linear penalty for violations of max/min active/reactive power from
    external grids. """

    upper_violations = calc_penalty(
        net, 'res_ext_grid', column, f'max_{column}', vectorize=vectorize)
    lower_violations = calc_penalty(
        net, 'res_ext_grid', column, f'min_{column}', False, vectorize=vectorize)

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
    if violations > 0:
        print(f'{column} power over max: ', violations * penalty_factor)

    if autocorrect:
        correct_active_reactive_overpower(net, column)

    return violations * penalty_factor


def correct_active_reactive_overpower(net, column):
    """ Active power is not automatically bounded by the agent. Invalid
    actions need to be ignored, if necessary. """
    new_values = np.minimum(
        net.sgen[column].to_numpy(), net.sgen[f'max_{column}'].to_numpy())
    net.sgen[column] = new_values
