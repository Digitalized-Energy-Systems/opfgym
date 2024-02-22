
import numpy as np


def compute_violation(net, unit_type: str, column: str, min_or_max: str):
    values = net['res_' + unit_type][column].to_numpy()
    boundary = net[unit_type][f'{min_or_max}_{column}']

    invalids = values > boundary if min_or_max == 'max' else values < boundary
    absolute_violations = (values - boundary)[invalids].abs()
    # TODO: This fails for boundary == 0 -> use alternative reference values?!
    percentage_violations = (absolute_violations / boundary[invalids]).abs() * 100

    return absolute_violations.to_numpy(), percentage_violations.to_numpy(), invalids


def compute_penalty(violation: float, n_violations: int, linear_penalty=0,
                    quadr_penalty=0, offset_penalty=0, sqrt_penalty=0):
    """ General function to compute linear, quadratic, anc offset penalties
    for constraint violations in pandapower nets """

    penalty = violation * linear_penalty
    penalty += violation**2 * quadr_penalty
    penalty += violation**0.5 * sqrt_penalty

    # Penalize every violation with constant factor
    penalty += n_violations * offset_penalty

    return -penalty


def voltage_violation(net, *args, **kwargs):
    """ Penalty for voltage violations of the upper or lower voltage
    boundary (both treated equally). """
    violations1, perc_violations1, invalids1 = compute_violation(
        net, 'bus', 'vm_pu', 'max')
    violations2, perc_violations2,  invalids2 = compute_violation(
        net, 'bus', 'vm_pu', 'min')

    violation = violations1.sum() + violations2.sum()
    percentage_violation = perc_violations1.sum() + perc_violations2.sum()
    invalids = np.logical_or(invalids1, invalids2)
    penalty = compute_penalty(violation, sum(invalids), *args, **kwargs)

    return ~invalids.any(), violation, percentage_violation, penalty


def line_overload(net, *args, **kwargs):
    """ Penalty for overloaded lines. Only max boundary required! """
    violation, perc_violation, invalids = compute_violation(
        net, 'line', 'loading_percent', 'max')
    penalty = compute_penalty(sum(violation), sum(invalids), *args, **kwargs)
    return ~invalids.any(), sum(violation), sum(perc_violation), penalty


def trafo_overload(net, *args, **kwargs):
    """ Penalty for overloaded trafos. Only max boundary required! """
    violation, perc_violation, invalids = compute_violation(
        net, 'trafo', 'loading_percent', 'max')
    penalty = compute_penalty(sum(violation), sum(invalids), *args, **kwargs)
    return ~invalids.any(), sum(violation), sum(perc_violation), penalty


def ext_grid_overpower(net, column='p_mw', *args, **kwargs):
    """ Penalty for violations of max/min active/reactive power from
    external grids. """
    violations1, perc_violations1, invalids1 = compute_violation(
        net, 'ext_grid', column, 'max')
    violations2, perc_violations2, invalids2 = compute_violation(
        net, 'ext_grid', column, 'min')

    violation = violations1.sum() + violations2.sum()
    percentage_violation = perc_violations1.sum() + perc_violations2.sum()
    invalids = np.logical_or(invalids1, invalids2)
    penalty = compute_penalty(violation, sum(invalids), *args, **kwargs)
    
    return ~invalids.any(), violation, percentage_violation, penalty
