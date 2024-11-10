import pytest

import numpy as np
import pandapower as pp
import pandapower.networks as pn

import opfgym.objective as objective


@pytest.fixture
def net():
    net = pn.example_simple()
    pp.runpp(net)
    return net


def test_get_powers_from_pwl_cost(net):
    pp.create_pwl_cost(net, 0, 'load', power_type='p', points=[[0, 1, 30], [1, 2, 50]])
    pp.create_pwl_cost(net, 0, 'load', power_type='q', points=[[0, 1, 30], [1, 2, 50]])
    net.res_load.loc[0, 'p_mw'] = 1.5
    net.res_load.loc[0, 'q_mvar'] = 2.0
    assert (np.array(objective.get_powers_from_pwl_cost(net)) == np.array([1.5, 2.0])).all()

    pp.create_pwl_cost(net, 0, 'sgen', power_type='p', points=[[0, 1, 30], [1, 2, 50]])
    pp.create_pwl_cost(net, 0, 'gen', power_type='p', points=[[0, 1, 30], [1, 2, 50]])
    net.res_sgen.loc[0, 'p_mw'] = 1.2
    net.res_gen.loc[0, 'p_mw'] = 1.4
    assert (np.array(objective.get_powers_from_pwl_cost(net)) == np.array([1.5, 2.0, 1.2, 1.4])).all()

def test_get_piecewise_linear_costs(net):
    pp.create_pwl_cost(net, 0, 'load', power_type='p', points=[[0, 1, 30], [1, 2, 50]])
    net.res_load.loc[0, 'p_mw'] = 1.5
    assert np.sum(objective.get_piecewise_linear_costs(net)) == 30 + 25

    pp.create_pwl_cost(net, 0, 'load', power_type='q', points=[[0, 1, 30], [1, 2, 50]])
    net.res_load.loc[0, 'q_mvar'] = 2.0
    assert np.sum(objective.get_piecewise_linear_costs(net)) == 30 + 25 + 30 + 50

    pp.create_pwl_cost(net, 0, 'gen', power_type='p', points=[[0, 1, 30], [1, 2, 50]])
    net.res_gen.loc[0, 'p_mw'] = 0.5
    assert np.sum(objective.get_piecewise_linear_costs(net)) == 30 + 25 + 30 + 50 + 15

    pp.create_pwl_cost(net, 0, 'gen', power_type='q', points=[[-1, 0, 40], [0, 1, 30], [1, 2, 50]])
    net.res_gen.loc[0, 'q_mvar'] = -0.5
    assert np.sum(objective.get_piecewise_linear_costs(net)) == -20 + 30 + 25 + 30 + 50 + 15

    pp.create_pwl_cost(net, 0, 'sgen', power_type='p', points=[[0, 1, 30], [1, 2, 50]])
    net.res_sgen.loc[0, 'p_mw'] = -0.5
    # No price defined for negative power -> same costs as before
    # While this is expected behaviour, it is even better to not allow powers with undefined prices
    assert np.sum(objective.get_piecewise_linear_costs(net)) == -20 + 30 + 25 + 30 + 50 + 15

def test_get_powers_from_poly_cost(net):
    pp.create_poly_cost(net, 0, 'load', cp1_eur_per_mw=2)
    net.res_load.loc[0, 'p_mw'] = 1.5
    net.res_load.loc[0, 'q_mvar'] = 1.8
    assert (objective.get_powers_from_poly_cost(net, 'p_mw') == np.array([1.5])).all()

    pp.create_poly_cost(net, 0, 'sgen', cp1_eur_per_mw=2, cq1_eur_per_mvar=2)
    net.res_sgen.loc[0, 'p_mw'] = 1.2
    net.res_sgen.loc[0, 'q_mvar'] = 2.0
    assert (objective.get_powers_from_poly_cost(net, 'q_mvar') == np.array([1.8, 2.0])).all()
    assert (objective.get_powers_from_poly_cost(net, 'p_mw') == np.array([1.5, 1.2])).all()

def test_get_polynomial_costs(net):
    pp.create_poly_cost(net, 0, 'load', cp1_eur_per_mw=2)
    net.res_load.loc[0, 'p_mw'] = 1.5
    net.res_load.loc[0, 'q_mvar'] = 2.0
    assert np.sum(objective.get_polynomial_costs(net)) == 3

    pp.create_poly_cost(net, 0, 'sgen', cp1_eur_per_mw=2, cq1_eur_per_mvar=2)
    net.res_sgen.loc[0, 'p_mw'] = 1.2
    net.res_sgen.loc[0, 'q_mvar'] = 2.0
    assert (objective.get_polynomial_costs(net) == np.array([3.0, 2.4, 0, 4.0])).all()

    net.poly_cost.loc[0, 'cp0_eur'] = 1
    net.poly_cost.loc[1, 'cq2_eur_per_mvar2'] = 2
    assert (objective.get_polynomial_costs(net) == np.array([4.0, 2.4, 0, 4.0+8.0])).all()

def test_get_pandapower_costs(net):
    pp.create_poly_cost(net, 0, 'sgen', cp1_eur_per_mw=30)
    net.res_sgen.loc[0, 'p_mw'] = 2
    assert (objective.get_pandapower_costs(net) == np.array([60, 0])).all()

    pp.create_pwl_cost(net, 0, 'load', power_type='p', points=[[0, 1, 30], [1, 2, 50]])
    net.res_load.loc[0, 'p_mw'] = 1.5
    print(objective.get_pandapower_costs(net))
    assert (objective.get_pandapower_costs(net) == np.array([60, 0, 30+25,])).all()
