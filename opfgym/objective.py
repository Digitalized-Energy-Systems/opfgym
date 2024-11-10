
import numpy as np
import pandas as pd


def get_pandapower_costs(net) -> np.ndarray:
    """
    Total costs as implemented in pandapower network as minimization problem.
    Useful if cost function is already implemented or for comparison with
    pandapower-OPF. Attention: Not always equivalent to 'net.res_cost' after
    pp-OPF, because internal cost calculation of pandapower is strange.

    Returns an array with length of 2*len(net.poly_cost) + len(net.pwl_cost)
    because poly_cost defines costs for active and(!) reactive power, while
    pwl_cost specificially defines costs for one of the two only. Sum up the
    cost array to get the total costs.
    """

    all_costs = []

    if len(net.poly_cost) > 0:
        all_costs.append(get_polynomial_costs(net))

    if len(net.pwl_cost) > 0:
        all_costs.append(get_piecewise_linear_costs(net))

    try:
        return np.concatenate(all_costs)
    except ValueError:
        # If no costs are defined, return empty array
        return np.array([])


def get_polynomial_costs(net) -> np.ndarray:
    p_mw = get_powers_from_poly_cost(net, 'p_mw')
    q_mvar = get_powers_from_poly_cost(net, 'q_mvar')

    p_costs = net.poly_cost.cp0_eur.copy()
    p_costs += net.poly_cost.cp1_eur_per_mw * p_mw
    p_costs += net.poly_cost.cp2_eur_per_mw2 * p_mw**2
    q_costs = net.poly_cost.cq0_eur.copy()
    q_costs += net.poly_cost.cq1_eur_per_mvar * q_mvar
    q_costs += net.poly_cost.cq2_eur_per_mvar2 * q_mvar**2

    return np.concatenate([p_costs, q_costs])


def get_powers_from_poly_cost(net, column: str) -> pd.Series:
    def extract_power_value(row):
        unit_type = f'res_{row["et"]}'
        power_idx = row['element']
        return net[unit_type].loc[power_idx, column]

    return net.poly_cost.apply(extract_power_value, axis=1)


def get_piecewise_linear_costs(net) -> np.ndarray:
    powers = get_powers_from_pwl_cost(net)
    costs = pd.Series(0, index=net.pwl_cost.index)
    for points in zip(*net.pwl_cost.points):
        lower, higher, price = map(np.array, zip(*points))

        signs = np.sign(powers)
        # Warning: Does not work if lower<0 and higher>0
        same_sign_flag = (signs == np.sign(lower + higher))
        lower_abs = np.abs(lower)
        higher_abs = np.abs(higher)
        power_abs = powers.abs()
        inside_abs = np.minimum(lower_abs, higher_abs)
        inside_flag = np.logical_and(power_abs > inside_abs, same_sign_flag)
        outside_flag = power_abs > np.maximum(lower_abs, higher_abs)
        intermediate_flag = np.logical_and(inside_flag, ~outside_flag)

        costs[outside_flag] += (signs * (higher - lower) * price)
        costs[intermediate_flag] += (signs * (power_abs - inside_abs) * price)

    return costs.to_numpy()


def get_powers_from_pwl_cost(net) -> pd.Series:
    def extract_power_value(row):
        unit_type = f'res_{row["et"]}'
        power_column = 'p_mw' if row['power_type'] == 'p' else 'q_mvar'
        power_idx = row['element']
        return net[unit_type].loc[power_idx, power_column]

    return net.pwl_cost.apply(extract_power_value, axis=1)
