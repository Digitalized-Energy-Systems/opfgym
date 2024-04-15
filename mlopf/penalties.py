
import numpy as np


def compute_total_violation(net, unit_type: str, column: str, min_or_max: str, 
                            worst_case_only=False, *args, **kwargs):
    values = net['res_' + unit_type][column].to_numpy()
    boundary = net[unit_type][f'{min_or_max}_{column}']
    if hasattr(net[unit_type], 'scaling') and column in ('p_mw', 'q_mvar'):
        # Constraints are applied to the scaled power values!
        boundary *= net[unit_type].scaling

    invalids = values > boundary if min_or_max == 'max' else values < boundary
    if invalids.sum() == 0:
        # No constraint violations  
        return 0, 0, 0

    absolute_violations = (values - boundary)[invalids].abs()
    # TODO: This fails for boundary == 0 -> use alternative reference values?!
    percentage_violations = (absolute_violations / boundary[invalids]).abs() * 100

    if worst_case_only:
        return absolute_violations.max(), percentage_violations.max(), sum(invalids)

    return absolute_violations.sum(), percentage_violations.sum(), sum(invalids)


def compute_penalty(violation: float, n_violations: int, linear_penalty=0,
                    quadr_penalty=0, offset_penalty=0, sqrt_penalty=0, 
                    *args, **kwargs):
    """ General function to compute linear, quadratic, anc offset penalties
    for constraint violations in pandapower nets """

    penalty = violation * linear_penalty
    # TODO: Should this really happen for the sum of violations? (**2 higher this way)
    penalty += violation**2 * quadr_penalty
    penalty += violation**0.5 * sqrt_penalty

    # Penalize every violation with constant factor
    penalty += n_violations * offset_penalty

    return -penalty


def voltage_violation(net, *args, **kwargs):
    """ Penalty for voltage violations of the upper or lower voltage
    boundary (both treated equally). """
    violation1, perc_violation1, n_invalids1 = compute_total_violation(
        net, 'bus', 'vm_pu', 'max', **kwargs)
    violations2, perc_violations2,  n_invalids2 = compute_total_violation(
        net, 'bus', 'vm_pu', 'min', **kwargs)

    violation = violation1 + violations2
    percentage_violation = perc_violation1 + perc_violations2
    n_invalids = n_invalids1 + n_invalids2
    penalty = compute_penalty(violation, n_invalids, *args, **kwargs)

    return not bool(n_invalids), violation, percentage_violation, penalty


def line_overload(net, *args, **kwargs):
    """ Penalty for overloaded lines. Only max boundary required! """
    violation, perc_violation, n_invalids = compute_total_violation(
        net, 'line', 'loading_percent', 'max', **kwargs)
    penalty = compute_penalty(violation, n_invalids, *args, **kwargs)
    return not bool(n_invalids), violation, perc_violation, penalty


def trafo_overload(net, *args, **kwargs):
    """ Penalty for overloaded trafos. Only max boundary required! """
    violation, perc_violation, n_invalids = compute_total_violation(
        net, 'trafo', 'loading_percent', 'max', **kwargs)
    penalty = compute_penalty(violation, n_invalids, *args, **kwargs)
    return not bool(n_invalids), violation, perc_violation, penalty


def ext_grid_overpower(net, column='p_mw', *args, **kwargs):
    """ Penalty for violations of max/min active/reactive power from
    external grids. """
    violation1, perc_violation1, n_invalids1 = compute_total_violation(
        net, 'ext_grid', column, 'max', **kwargs)
    violations2, perc_violations2, n_invalids2 = compute_total_violation(
        net, 'ext_grid', column, 'min', **kwargs)

    violation = violation1 + violations2
    percentage_violation = perc_violation1 + perc_violations2
    n_invalids = n_invalids1 + n_invalids2
    penalty = compute_penalty(violation, n_invalids, *args, **kwargs)
    
    return not bool(n_invalids), violation, percentage_violation, penalty
