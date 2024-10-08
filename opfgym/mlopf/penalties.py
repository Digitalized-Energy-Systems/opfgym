
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
        return 0, 0

    absolute_violations = (values - boundary)[invalids].abs()

    if worst_case_only:
        # Only return the single worst case violation
        return absolute_violations.max(), 1

    return absolute_violations.sum(), sum(invalids)


def compute_penalty(violation: float, n_violations: int, linear_penalty=1,
                    quadr_penalty=0, offset_penalty=0, sqrt_penalty=0, 
                    *args, **kwargs):
    """ General function to compute linear, quadratic, and offset penalties
    for constraint violations in pandapower nets """

    penalty = violation * linear_penalty
    # TODO: Should this really happen for the sum of violations? (**2 higher this way)
    penalty += violation**2 * quadr_penalty
    penalty += violation**0.5 * sqrt_penalty

    # Penalize every violation with constant factor
    penalty += n_violations * offset_penalty

    return -penalty


def voltage_violation(net, autoscale=False, *args, **kwargs):
    """ Penalty for voltage violations of the upper or lower voltage
    boundary (both treated equally). """
    violation1, n_invalids1 = compute_total_violation(
        net, 'bus', 'vm_pu', 'max', **kwargs)
    violation2,  n_invalids2 = compute_total_violation(
        net, 'bus', 'vm_pu', 'min', **kwargs)

    violation = violation1 + violation2
    n_invalids = n_invalids1 + n_invalids2

    if autoscale:
        # Scale violation to values around ~1 to make them comparable
        violation *= 20  # Assuming a typical voltage violation of 0.05 pu

    penalty = compute_penalty(violation, n_invalids, *args, **kwargs)

    return not bool(n_invalids), violation, penalty


def line_overload(net, autoscale=False, *args, **kwargs):
    """ Penalty for overloaded lines. Only max boundary required! """
    violation, n_invalids = compute_total_violation(
        net, 'line', 'loading_percent', 'max', **kwargs)

    if autoscale:
        # Scale violation to values around ~1 to make them comparable
        violation /= 30  # Assuming a typical line overload of 30%

    penalty = compute_penalty(violation, n_invalids, *args, **kwargs)

    return not bool(n_invalids), violation, penalty


def trafo_overload(net, autoscale=False, *args, **kwargs):
    """ Penalty for overloaded trafos. Only max boundary required! """
    violation, n_invalids = compute_total_violation(
        net, 'trafo', 'loading_percent', 'max', **kwargs)
    
    if autoscale:
        # Scale violation to values around ~1 to make them comparable
        violation /= 30  # Assuming a typical trafo overload of 30%

    penalty = compute_penalty(violation, n_invalids, *args, **kwargs)

    return not bool(n_invalids), violation, penalty


def ext_grid_overpower(net, column='p_mw', autoscale=False, *args, **kwargs):
    """ Penalty for violations of max/min active/reactive power from
    external grids. """
    violation1, n_invalids1 = compute_total_violation(
        net, 'ext_grid', column, 'max', **kwargs)
    violation2, n_invalids2 = compute_total_violation(
        net, 'ext_grid', column, 'min', **kwargs)

    violation = violation1 + violation2
    n_invalids = n_invalids1 + n_invalids2

    if autoscale:
        # Scale violation to values around ~1 to make them comparable
        # Use the load and sgen power as heuristic for scaling
        violation /= abs(net.ext_grid[f'mean_{column}'].sum())

    penalty = compute_penalty(violation, n_invalids, *args, **kwargs)
    
    return not bool(n_invalids), violation, penalty
