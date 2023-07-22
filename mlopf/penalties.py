from typing import Tuple

import numpy as np
from pandapower import pandapowerNet
from pandas import DataFrame


def compute_violation(net: pandapowerNet, unit_type: str, column: str, min_or_max: str) -> Tuple[np.ndarray, np.ndarray, DataFrame]:
    values: DataFrame = net['res_' + unit_type][column]
    boundary: DataFrame = net[unit_type][f'{min_or_max}_{column}']

    invalids: DataFrame = values > boundary if min_or_max == 'max' else values < boundary
    absolute_violations = (values - boundary)[invalids].abs()
    percentage_violations = (absolute_violations / boundary[invalids]).abs()

    return absolute_violations.to_numpy(), percentage_violations.to_numpy(), invalids


def compute_penalty(violation: float, n_violations: int, linear_penalty: int = 0, quadr_penalty: int = 0, offset_penalty: float = 0.0, sqrt_penalty: float = 0.0) -> float:
    """
    General function to compute linear, quadratic, anc offset penalties for constraint violations in pandapower nets.
    """

    penalty: float = violation * linear_penalty
    penalty += violation ** 2 * quadr_penalty
    penalty += violation ** 0.5 * sqrt_penalty

    # Penalize every violation with constant factor
    penalty += n_violations * offset_penalty

    return -penalty


def voltage_violation(net: pandapowerNet, *args, **kwargs) -> Tuple[bool, float, float, float]:
    """
    Penalty for voltage violations of the upper or lower voltage boundary (both treated equally).
    """

    violations1, perc_violations1, invalids1 = compute_violation(net, 'bus', 'vm_pu', 'max')
    violations2, perc_violations2, invalids2 = compute_violation(net, 'bus', 'vm_pu', 'min')

    violation: float = violations1.sum() + violations2.sum()
    percentage_violation: float = perc_violations1.sum() + perc_violations2.sum()
    invalids: np.ndarray = np.logical_or(invalids1.to_numpy(), invalids2.to_numpy())
    penalty: float = compute_penalty(violation, len(invalids), *args, **kwargs)

    return ~invalids.any(), violation, percentage_violation, penalty


def line_overload(net: pandapowerNet, *args, **kwargs) -> Tuple[bool, float, float, float]:
    """
    Penalty for overloaded lines. Only max boundary required!
    """
    violation, perc_violation, invalids = compute_violation(net, 'line', 'loading_percent', 'max')
    penalty: float = compute_penalty(violation.sum(), len(invalids), *args, **kwargs)
    return ~invalids.to_numpy().any(), violation.sum(), perc_violation.sum(), penalty


def trafo_overload(net, *args, **kwargs) -> Tuple[bool, float, float, float]:
    """
    Penalty for overloaded trafos. Only max boundary required!
    """
    violation, perc_violation, invalids = compute_violation(net, 'trafo', 'loading_percent', 'max')
    penalty: float = compute_penalty(violation.sum(), len(invalids), *args, **kwargs)
    return ~invalids.to_numpy().any(), violation.sum(), perc_violation.sum(), penalty


def ext_grid_overpower(net: pandapowerNet, column: str = 'p_mw', *args, **kwargs) -> Tuple[bool, float, float, float]:
    """
    Penalty for violations of max/min active/reactive power from external grids.
    """

    violations1, perc_violations1, invalids1 = compute_violation(net, 'ext_grid', column, 'max')
    violations2, perc_violations2, invalids2 = compute_violation(net, 'ext_grid', column, 'min')

    violation: float = violations1.sum() + violations2.sum()
    percentage_violation: float = perc_violations1.sum() + perc_violations2.sum()
    invalids: np.ndarray = np.logical_or(invalids1.to_numpy(), invalids2.to_numpy())
    penalty: float = compute_penalty(violation, len(invalids), *args, **kwargs)

    return ~invalids.any(), violation, percentage_violation, penalty
