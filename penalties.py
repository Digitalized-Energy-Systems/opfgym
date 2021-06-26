import numpy as np

# TODO: Lots of code repetition here -> one function that gets called by all!


def voltage_violation(net, penalty):
    """ Linear penalty for voltage violations of the upper or lower voltage
    boundary (both treated equally). """
    voltages = net.res_bus.vm_pu.to_numpy()
    max_voltages = net.bus["max_vm_pu"].to_numpy()
    min_voltages = net.bus["min_vm_pu"].to_numpy()

    upper_mask = voltages > max_voltages
    lower_mask = voltages < min_voltages

    upper_violations = (voltages - max_voltages)[upper_mask].sum()
    lower_violations = (min_voltages - voltages)[lower_mask].sum()
    # TODO: use vector instead
    penalty = (upper_violations + lower_violations) * penalty
    if upper_violations > 0:
        print('overvoltage: ', upper_violations * penalty)
    if lower_violations > 0:
        print('undervoltage: ', lower_violations * penalty)
    print(voltages)
    return penalty


def line_overload(net, penalty):
    line_loads = net.res_line.loading_percent.to_numpy()
    max_line_loads = net.line.max_loading_percent.to_numpy()

    mask = line_loads > max_line_loads
    violations = (line_loads - max_line_loads)[mask].sum()
    if violations > 0:
        print('line overload: ', violations * penalty)
    return violations * penalty


def apparent_overpower(net, penalty, autocorrect=True):
    power = (net.sgen.p_mw.to_numpy() ** 2 +
             net.sgen.q_mvar.to_numpy() ** 2)**0.5
    max_power = net.sgen.max_s_mva.to_numpy()

    mask = power > max_power
    violations = (power - max_power)[mask].sum()
    if violations > 0:
        print('apparent power over max: ', violations * penalty)

    if autocorrect:
        correct_overpower(net)

    return violations * penalty


def correct_overpower(net):
    """ Apparent power is not automatically bounded by the agent. Invalid
    actions need to be ignored, if necessary. Assumption: Always reduce
    reactive power, if apparent power is too high. """
    s_mva2 = net.sgen.max_s_mva.to_numpy() ** 2
    p_mw2 = net.sgen.p_mw.to_numpy() ** 2
    q_mvar_max = (s_mva2 - p_mw2)**0.5
    new_values = np.minimum(net.sgen['q_mvar'].abs(), q_mvar_max)
    net.sgen['q_mvar'] = np.sign(net.sgen['q_mvar']) * new_values
