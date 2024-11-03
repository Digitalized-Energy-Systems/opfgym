"""
Defines a general class to define new constraints in an RL-OPF environment.

Also, pre-implements the standard pandapower constraints VoltageConstraint,
LineOverloadConstraint, TrafoOverloadConstraint, ExtGridActivePowerConstraint,
ExtGridReactivePowerConstraint, which are compatible with the pandapower OPF.

Also, provides a function to extract the default constraints from a pandapower
network.
"""

import numpy as np
import pandas as pd


class Constraint():
    """ General class to define constraints in the RL-OPF environment.

    Warning: The utilized pandapower network requires a column of constrained
    values and one or two more columns that define the max/min boundary of
    these values. For example, values_column='s_mva' with a defined 'max_s_mva'
    in the same table.

    Attributes:
        unit_type: Which pandapower table to look at for constraint data.
        values_column : Which column in the previous table?
        only_worst_case_violations: Only compute the worst-case violations? (instead of the sum of all)
        auto_scale_violation: Scale violations to be of similar magnitude?
        scale_bounded_values: Apply the scaling column to the bounded values? Required if the constraint is scaled as well, for example, for apparent power s_mva)
        penalty_factor: Penalty scaling factor
        penalty_power: Power to apply to the violation (e.g. 0.5 for square root, 2 for quadratic, etc.)
        violation_count_penalty: Penalty per count of violation
    """
    def __init__(self,
                 unit_type: str,
                 values_column: str,
                 only_worst_case_violations: bool=False,
                 autoscale_violation: bool=True,
                 scale_bounded_values: bool=False,
                 penalty_factor: float=1.0,
                 penalty_power: float=1.0,
                 violation_count_penalty: float=0.0):
        self.unit_type = unit_type
        self.values_column = values_column
        self.only_worst_case_violations = only_worst_case_violations
        self.autoscale_violation = autoscale_violation
        self.scale_bounded_values = scale_bounded_values

        self.penalty_factor = penalty_factor
        self.penalty_power = penalty_power
        self.violation_count_penalty = violation_count_penalty

    def __call__(self, net) -> dict:
        return self.get_violation_metrics(net)

    def get_violation_metrics(self, net) -> dict:
        values = self.get_bounded_values(net)
        boundaries = self.get_boundaries(net)

        violation = 0
        penalty = 0
        n_violations = 0
        for min_or_max, boundary in boundaries.items():
            invalids = self.get_invalid_flag(values, boundary, min_or_max)
            n_violations += invalids.sum()
            violation += self.calculate_violation(values, boundary, invalids)

        if self.autoscale_violation:
            violation *= self.autoscale_violation

        penalty += self.calculate_penalty(violation, n_violations)
        valid = (n_violations == 0)

        return {'valid': valid, 'violation': violation, 'penalty': penalty}

    def get_bounded_values(self, net):
        return net['res_' + self.unit_type][self.values_column]

    def get_boundaries(self, net):
        return {
            min_or_max: self.get_single_boundary(net, min_or_max)
            for min_or_max in ('min', 'max')
            if f'{min_or_max}_{self.values_column}' in net[self.unit_type]
        }

    def get_single_boundary(self, net, min_or_max):
        boundary = net[self.unit_type][f'{min_or_max}_{self.values_column}']
        return self.scale_boundary(net, boundary)

    def scale_boundary(self, net, boundary):
        if self.scale_bounded_values or ('scaling' in net[self.unit_type]
                and self.values_column in ('p_mw', 'q_mvar')):
            return boundary * net[self.unit_type].scaling
        return boundary

    def get_invalid_flag(self, values, boundary, min_or_max) -> pd.Series:
        return values > boundary if min_or_max == 'max' else values < boundary

    def calculate_violation(self, values, boundary, invalids) -> np.ndarray:
        if invalids.sum() == 0:
            return 0

        absolute_violations = (values - boundary)[invalids].abs()

        if self.only_worst_case_violations:
            return absolute_violations.max()

        return absolute_violations.sum()

    def calculate_penalty(self, violation: float, n_violations: int) -> float:
        penalty = violation**(self.penalty_power) * self.penalty_factor
        penalty += n_violations * self.violation_count_penalty

        return -penalty


class VoltageConstraint(Constraint):
    def __init__(self, autoscale_violation=True, **args):
        if autoscale_violation is True:
            # pu values are typically very small around 0.05-0.1 -> scale higher
            autoscale_violation = 20
        super().__init__(unit_type='bus',
                         values_column='vm_pu',
                         autoscale_violation=autoscale_violation,
                         **args)


class LineOverloadConstraint(Constraint):
    def __init__(self, autoscale_violation=True, **args):
        if autoscale_violation is True:
            # overload values are typically around 10-30 -> scale lower
            autoscale_violation = 1/30
        super().__init__(unit_type='line',
                         values_column='loading_percent',
                         autoscale_violation=autoscale_violation,
                         **args)


class TrafoOverloadConstraint(Constraint):
    def __init__(self, autoscale_violation=True, **args):
        if autoscale_violation is True:
            # overload values are typically around 10-30 -> scale lower
            autoscale_violation = 1/30
        super().__init__(unit_type='trafo',
                         values_column='loading_percent',
                         autoscale_violation=autoscale_violation,
                         **args)


class Trafo3wOverloadConstraint(Constraint):
    def __init__(self, autoscale_violation=True, **args):
        if autoscale_violation is True:
            # overload values are typically around 10-30 -> scale lower
            autoscale_violation = 1/30
        super().__init__(unit_type='trafo3w',
                         values_column='loading_percent',
                         autoscale_violation=autoscale_violation,
                         **args)


class ExtGridActivePowerConstraint(Constraint):
    def __init__(self, **args):
        super().__init__(unit_type='ext_grid', values_column='p_mw', **args)

    def get_violation_metrics(self, net) -> dict:
        if not self.autoscale_violation:
            self.autoscale_violation = 1 / abs(net.ext_grid['mean_p_mw'].sum())
        return super().get_violation_metrics(net)


class ExtGridReactivePowerConstraint(Constraint):
    def __init__(self, **args):
        super().__init__(unit_type='ext_grid', values_column='q_mvar', **args)

    def get_violation_metrics(self, net) -> dict:
        if not self.autoscale_violation:
            self.autoscale_violation = 1 / abs(net.ext_grid['mean_q_mvar'].sum())
        return super().get_violation_metrics(net)


def create_default_constraints(net, constraint_kwargs: dict) -> list:
    """ Extract and return default constraints from the pandapower network if
    defined there.
    (compare https://pandapower.readthedocs.io/en/latest/opf/formulation.html)
    """
    constraints = []

    max_vm_pu_defined = is_constraint_defined(net, 'bus', 'max_vm_pu')
    min_vm_pu_defined = is_constraint_defined(net, 'bus', 'min_vm_pu')
    if max_vm_pu_defined or min_vm_pu_defined:
        constraints.append(VoltageConstraint(**constraint_kwargs))

    if is_constraint_defined(net, 'line', 'max_loading_percent'):
        constraints.append(LineOverloadConstraint(**constraint_kwargs))

    if is_constraint_defined(net, 'trafo', 'max_loading_percent'):
        constraints.append(TrafoOverloadConstraint(**constraint_kwargs))

    if is_constraint_defined(net, 'trafo3w', 'max_loading_percent'):
        constraints.append(Trafo3wOverloadConstraint(**constraint_kwargs))

    max_p_mw_defined = is_constraint_defined(net, 'ext_grid', 'max_p_mw')
    min_p_mw_defined = is_constraint_defined(net, 'ext_grid', 'min_p_mw')
    if max_p_mw_defined or min_p_mw_defined:
        constraints.append(ExtGridActivePowerConstraint(**constraint_kwargs))

    max_q_mvar_defined = is_constraint_defined(net, 'ext_grid', 'max_q_mvar')
    min_q_mvar_defined = is_constraint_defined(net, 'ext_grid', 'min_q_mvar')
    if max_q_mvar_defined or min_q_mvar_defined:
        constraints.append(ExtGridReactivePowerConstraint(**constraint_kwargs))

    return constraints


def is_constraint_defined(net, unit_type: str, constraint_column: str) -> bool:
    return (constraint_column in net[unit_type]
            and has_numeric_finite_value(net[unit_type][constraint_column]))


def has_numeric_finite_value(series: pd.Series) -> bool:
    # Set errors='coerce' to convert non-numeric to NaN
    numeric_series = pd.to_numeric(series, errors='coerce')
    # Check if at least one value is a finite number (not NaN or Inf)
    return np.isfinite(numeric_series).any()
