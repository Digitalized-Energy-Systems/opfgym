
import numpy as np

from opfgym import OpfEnv


class SecurityConstrainedOpfEnv(OpfEnv):
    """ Environment class for security-constrained OPF environments. This class
    implements the possibility to remove n-1 elements from the network by
    implementing a wrapper around the calculate_violations method of the base
    class.

    Args:
        n_minus_one_keys: Tuple of tuples of strings and numpy arrays. Each
            tuple should contain the following elements:
            - unit_type: Type of unit to be removed (e.g. 'line', 'trafo')
            - column: Column to be set to False (e.g. 'in_service', 'closed')
            - idxs: Numpy array with the indices of the elements to be removed.
        not_converged_penalty: Penalty to be applied when the power flow does
            not converge in the contingency case. Defaults to 1.
        **kwargs: Keyword arguments to be passed to the base class.
    """
    def __init__(
            self,
            *args,
            n_minus_one_keys: tuple[tuple[str, str, np.ndarray], ...],
            not_converged_penalty: float = 1,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.not_converged_penalty = not_converged_penalty

        self.n_minus_one_keys = n_minus_one_keys
        for unit_type, column, idxs in self.n_minus_one_keys:
            assert column in ('in_service', 'closed')

    def calculate_violations(self, net=None):
        """ Wrapper around the original method. Implement the
        security-constrained OPF by removing the n-1 elements and summing all
        resulting violations. """
        net = net or self.net
        valids, viol, penalties = super().calculate_violations(net)

        for unit_type, column, idxs in self.n_minus_one_keys:
            for idx in idxs:
                cell_value = get_cell_value(net, unit_type, column, idx)
                if cell_value is False:
                    continue

                net = set_cell_to_false(net, unit_type, column, idx)

                try:
                    self._run_power_flow(net)
                    new_results = super().calculate_violations(net)
                    new_valids, new_violations, new_penalties = new_results
                    valids = np.logical_and(valids, new_valids)
                    viol += new_violations
                    penalties += new_penalties
                except:
                    # If power flow fails in any way, assume invalid
                    # state and use default penalty
                    valids = np.zeros_like(valids)
                    viol += self.not_converged_penalty
                    penalties += self.not_converged_penalty

                net = set_cell_to_true(net, unit_type, column, idx)

        return valids, viol, penalties


def get_cell_value(net, unit_type, column, idx):
    return net[unit_type].at[idx, column]


def set_cell_to_true(net, unit_type, column, idx):
    net[unit_type].at[idx, column] = True
    return net


def set_cell_to_false(net, unit_type, column, idx):
    """ Create contingency by setting a cell to False (usually in_service). """
    net[unit_type].at[idx, column] = False
    return net