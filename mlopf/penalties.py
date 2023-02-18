
def compute_penalty(net, unit_type: str, column: str, min_or_max: str,
                    linear_penalty=0, quadr_penalty=0, offset_penalty=0):
    """ General function to compute linear, quadratic, anc offset penalties
    for constraint violations in pandapower nets """

    values = net['res_' + unit_type][column].to_numpy()
    boundary = net[unit_type][f'{min_or_max}_{column}']

    mask = values > boundary if min_or_max == 'max' else values < boundary
    violations = (values - boundary)[mask].abs()

    penalty = (violations.sum()) * linear_penalty
    penalty += (violations**2).sum() * quadr_penalty

    # Penalize every violation with constant factor
    penalty += len(mask) * offset_penalty

    return penalty


def voltage_violation(net, *args, **kwargs):
    """ Default penalty for voltage violations of the upper or lower voltage
    boundary (both treated equally). """
    return (compute_penalty(net, 'bus', 'vm_pu', 'max', *args, **kwargs)
            + compute_penalty(net, 'bus', 'vm_pu', 'min', *args, **kwargs))


def line_overload(net, *args, **kwargs):
    """ Penalty for overloaded lines. Only max boundary required! """
    return compute_penalty(net, 'line', 'loading_percent', 'max', *args, **kwargs)


def trafo_overload(net, *args, **kwargs):
    """ Penalty for overloaded trafos. Only max boundary required! """
    return compute_penalty(net, 'trafo', 'loading_percent', 'max', *args, **kwargs)


def ext_grid_overpower(net, column='p_mw', *args, **kwargs):
    """ Penalty for violations of max/min active/reactive power from
    external grids. """
    return (compute_penalty(net, 'ext_grid', column, 'max', *args, **kwargs)
            + compute_penalty(net, 'ext_grid', column, 'min', *args, **kwargs))
