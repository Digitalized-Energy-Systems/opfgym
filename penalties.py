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
    return (upper_violations + lower_violations) * penalty


def line_overload(net, penalty):
    line_loads = net.res_line.loading_percent.to_numpy()
    max_line_loads = net.line.max_loading_percent.to_numpy()

    mask = line_loads > max_line_loads
    violations = (line_loads - max_line_loads)[mask].sum()
    return violations * penalty


def apparent_overpower(net, penalty):
    power = (net.sgen.p_mw.to_numpy() ** 2 +
             net.sgen.q_mvar.to_numpy() ** 2)**0.5
    max_power = net.sgen.max_s_mva.to_numpy()

    mask = power > max_power
    violations = (power - max_power)[mask].sum()
    return violations * penalty
