import pytest

import pandapower as pp
import pandapower.networks as pn

import opfgym.penalties as penalties


@pytest.fixture
def net():
    net = pn.example_simple()
    # Define constraints
    net.bus['max_vm_pu'] = 1.05
    net.bus['min_vm_pu'] = 0.95
    net.line['max_loading_percent'] = 100
    net.trafo['max_loading_percent'] = 100
    net.ext_grid['min_p_mw'] = -1
    net.ext_grid['max_p_mw'] = 1
    net.ext_grid['min_q_mvar'] = -1
    net.ext_grid['max_q_mvar'] = 1
    pp.runpp(net)
    return net


def test_voltage_violation(net):
    net.res_bus.vm_pu[1] = 1.1
    valid, violation, penalty = penalties.voltage_violation(
        net, linear_penalty=2)
    assert not valid
    assert round(violation, 3) == 0.05
    assert round(penalty, 3) == -0.1


def test_line_overloading(net):
    net.res_line.loading_percent[1] = 120
    valid, violation, penalty = penalties.line_overload(
        net, linear_penalty=2)
    assert not valid
    assert violation == 20
    assert penalty == -40.0


def test_trafo_overloading(net):
    net.res_trafo.loading_percent[0] = 130
    valid, violation, penalty = penalties.trafo_overload(
        net, linear_penalty=2)
    assert not valid
    assert violation == 30
    assert penalty == -60.0


def test_ext_grid_overpower(net):
    net.res_ext_grid.q_mvar[0] = 2
    valid, violation, penalty = penalties.ext_grid_overpower(
        net, column='q_mvar', linear_penalty=2)
    assert not valid
    assert violation == 1
    assert penalty == -2.0


def test_compute_penalty():
    violation = 10
    n_violations = 2
    penalty = penalties.compute_penalty(
        violation, n_violations, linear_penalty=3)
    assert penalty == -30

    penalty = penalties.compute_penalty(
        violation, n_violations, linear_penalty=0, offset_penalty=1.5)
    assert penalty == -3

    penalty = penalties.compute_penalty(
        violation, n_violations, offset_penalty=1.5, linear_penalty=2)
    assert penalty == -23


def test_compute_violation(net):
    net.res_line.loading_percent = 120
    net.line.max_loading_percent = 100

    violation, n_invalids = penalties.compute_total_violation(
        net, 'line', 'loading_percent', 'max')
    assert violation == 20 * len(net.line)

    violation, n_invalids = penalties.compute_total_violation(
        net, 'line', 'loading_percent', 'max', worst_case_only=True)
    assert violation == 20
