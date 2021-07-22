
"""
A set of objective functions for pandapower networks.
"""

import pandapower as pp


def min_p_loss(net):
    """ Minimize active power losses for a given network. """
    gen = (sum(net.res_ext_grid.p_mw)
           + sum(net.res_sgen.p_mw)
           + sum(net.res_gen.p_mw))
    load = sum(net.storage.p_mw) + sum(net.load.p_mw)
    p_loss = gen - load

    return p_loss


def max_p_feedin(net):
    """ Maximize active power feed-in of all generators. Negative sign
    necessary, because all objective functions must be min problems. """
    return -(sum(net.res_sgen.p_mw) + sum(net.res_gen.p_mw))


def min_v2_deviations(net):
    """ Minimize quadratic voltage deviations from reference voltage
    (1 pu). """
    return sum((net.res_bus.vm_pu - 1)**2)


def maximize_profit(net, units: dict):
    """ Maximize profit for some selected units in the network.
    All profits are added together!
    Units: dict of sets e.g. {'sgen': {1, 2, 3}} for sgens 1-3
    """
    profit = 0
    for unit_type, idxs in units.items():
        # TODO: Simplify by not iterating over all units!
        for unit_idx in idxs:
            p_mw = net[unit_type].p_mw[unit_idx]
            q_mvar = net[unit_type].q_mvar[unit_idx]
            costs = net.poly_cost[net.poly_cost.element ==
                                  unit_idx][net.poly_cost.et == unit_type]
            profit += sum(costs['cp0_eur'])
            profit += sum(costs['cp1_eur_per_mw'] * p_mw)
            profit += sum(costs['cp2_eur_per_mw2'] * p_mw**2)
            profit += sum(costs['cq0_eur'])
            profit += sum(costs['cq1_eur_per_mvar'] * q_mvar)
            profit += sum(costs['cq2_eur_per_mvar2'] * q_mvar**2)

    return profit


def min_pp_costs(net):
    """ Minimize total costs as implemented in pandapower network.
    Useful if cost function is already implemented or for comparison with
    pandapower-OPF. Attention: Not equivalent to 'net.res_cost' after
    pp-OPF, because internal cost calculation of pandapower is strange. """

    # TODO: piece-wise costs not implemented yet!
    costs = 0
    for idx in net.poly_cost.index:
        element = net.poly_cost.element[idx]
        et = net.poly_cost.et[idx]

        costs += net.poly_cost.cp0_eur[idx] + net.poly_cost.cq0_eur[idx]
        costs += (net[f'res_{et}']['p_mw'][element]
                  * net.poly_cost['cp1_eur_per_mw'][idx])
        costs += (net[f'res_{et}']['p_mw'][element]**2
                  * net.poly_cost['cp2_eur_per_mw2'][idx])
        costs += (net[f'res_{et}']['q_mvar'][element]
                  * net.poly_cost['cq1_eur_per_mvar'][idx])
        costs += (net[f'res_{et}']['q_mvar'][element]**2
                  * net.poly_cost['cq2_eur_per_mvar2'][idx])

    return costs


def add_min_loss_costs(net, p_price=30):
    """ Add polynomal costs to a pandapower network so that OPF minimizes active
    power losses. """
    for unit_type in ('sgen', 'gen', 'ext_grid', 'load', 'storage'):
        price = p_price
        if unit_type == 'load' or unit_type == 'storage':
            price = -p_price

        for idx in net[unit_type].index:
            pp.create_poly_cost(net, idx, unit_type, cp1_eur_per_mw=price)


def add_max_p_feedin(net, p_price=30):
    """ Add polynomal costs to a pandapower network so that OPF maximizes active
    power feed-in of all static generators (max renewable feed-in).
    The "p_price" is only relevant, if other objective fcts need to be
    considered. The price needs to be positive! """
    unit_type = 'sgen'
    for idx in net[unit_type].index:
        pp.create_poly_cost(net, idx, unit_type, cp1_eur_per_mw=-p_price)
