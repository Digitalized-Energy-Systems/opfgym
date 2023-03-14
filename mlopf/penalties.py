
def compute_penalty(net, unit_type: str, column: str, min_or_max: str,
                    info: dict, linear_penalty=0, quadr_penalty=0,
                    offset_penalty=0, sqrt_penalty=0):
    """ General function to compute linear, quadratic, anc offset penalties
    for constraint violations in pandapower nets """

    values = net['res_' + unit_type][column].to_numpy()
    boundary = net[unit_type][f'{min_or_max}_{column}']

    mask = values > boundary if min_or_max == 'max' else values < boundary
    if not mask.any():
        # No constraint violations -> no penalty + valid solution
        if f'violations_{unit_type}_{column}' not in info:
            info[f'violations_{unit_type}_{column}'] = 0.0
        return 0.0, True

    violations = (values - boundary)[mask].abs()

    if f'violations_{unit_type}_{column}' in info:
        info[f'violations_{unit_type}_{column}'] += violations.sum()
    else:
        info[f'violations_{unit_type}_{column}'] = violations.sum()
    # TODO: Store valids in `info`, too?!

    penalty = (violations.sum()) * linear_penalty
    penalty += (violations**2).sum() * quadr_penalty
    penalty += (violations**0.5).sum() * sqrt_penalty

    # Penalize every violation with constant factor
    penalty += len(mask) * offset_penalty

    return -penalty, False


def voltage_violation(net, info: dict, *args, **kwargs):
    """ Default penalty for voltage violations of the upper or lower voltage
    boundary (both treated equally). """
    pen1, valid1 = compute_penalty(
        net, 'bus', 'vm_pu', 'max', info, *args, **kwargs)
    pen2, valid2 = compute_penalty(
        net, 'bus', 'vm_pu', 'min', info, *args, **kwargs)
    return pen1 + pen2, valid1 and valid2


def line_overload(net, info: dict, *args, **kwargs):
    """ Penalty for overloaded lines. Only max boundary required! """
    return compute_penalty(
        net, 'line', 'loading_percent', 'max', info, *args, **kwargs)


def trafo_overload(net, info: dict, *args, **kwargs):
    """ Penalty for overloaded trafos. Only max boundary required! """
    return compute_penalty(
        net, 'trafo', 'loading_percent', 'max', info, *args, **kwargs)


def ext_grid_overpower(net, info: dict, column='p_mw', *args, **kwargs):
    """ Penalty for violations of max/min active/reactive power from
    external grids. """
    pen1, valid1 = compute_penalty(
        net, 'ext_grid', column, 'max', info, *args, **kwargs)
    pen2, valid2 = compute_penalty(
        net, 'ext_grid', column, 'min', info, *args, **kwargs)

    return pen1 + pen2, valid1 and valid2
