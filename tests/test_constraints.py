
import pytest

import numpy as np
import pandapower as pp
import pandapower.networks as pn

import opfgym.constraints as constraints


@pytest.fixture
def net():
    net = pn.example_simple()
    pp.runpp(net)
    return net

def test_voltage_constraint(net):
    constraint = constraints.VoltageConstraint(autoscale_violation=False,
                                               only_worst_case_violations=True)

    net['bus']['min_vm_pu'] = 0.95
    net['bus']['max_vm_pu'] = 1.05
    net.res_bus['vm_pu'] = 1.0
    net.res_bus.at[0, 'vm_pu'] = 0.9
    net.res_bus.at[1, 'vm_pu'] = 0.94  # Should be ignored for violation

    result = constraint(net)
    assert not result['valid']
    assert np.isclose(result['violation'], 0.05)  # max([0.95-0.9, 0.95-0.94])
    assert np.isclose(result['penalty'], -0.05)  # 0.05*1

def test_line_overload_constraint(net):
    constraint = constraints.LineOverloadConstraint(autoscale_violation=False,
                                                    penalty_factor=2.0)

    net['line']['max_loading_percent'] = 100
    net.res_line['loading_percent'] = 50
    net.res_line.at[0, 'loading_percent'] = 110

    result = constraint(net)
    assert not result['valid']
    assert result['violation'] == 10  # 110-100
    assert result['penalty'] == -20  # 10*2

def test_trafo_overload_constraint(net):
    constraint = constraints.TrafoOverloadConstraint(autoscale_violation=False,
                                                     penalty_power=2.0)

    net['trafo']['max_loading_percent'] = 100
    net.res_trafo['loading_percent'] = 50
    net.res_trafo.at[0, 'loading_percent'] = 110

    result = constraint(net)
    assert not result['valid']
    assert result['violation'] == 10  # 110-100
    assert result['penalty'] == -100  # 100**2

def test_ext_grid_active_power_constraint(net):
    constraint = constraints.ExtGridActivePowerConstraint(autoscale_violation=0.5)

    net['ext_grid']['min_p_mw'] = 0
    net.res_ext_grid.at[0, 'p_mw'] = -0.5

    result = constraint(net)
    assert not result['valid']
    assert result['violation'] == 0.25
    assert result['penalty'] == -0.25

def test_ext_grid_reactive_power_constraint(net):
    constraint = constraints.ExtGridReactivePowerConstraint(autoscale_violation=0.5)

    net['ext_grid']['min_q_mvar'] = 0
    net.res_ext_grid.at[0, 'q_mvar'] = -0.5

    result = constraint(net)
    assert not result['valid']
    assert result['violation'] == 0.25  # (0.5-0) * 0.5
    assert result['penalty'] == -0.25

def test_create_default_constraints(net):
    # This specific pandapower network has no constraints by default
    extracted_constraints = constraints.create_default_constraints(net, {})
    assert len(extracted_constraints) == 0

    # Add Voltage constraint
    net['bus']['min_vm_pu'] = 0.95
    net['bus']['max_vm_pu'] = 1.05
    extracted_constraints = constraints.create_default_constraints(net, {})
    assert len(extracted_constraints) == 1
    assert constraints.VoltageConstraint in [type(c) for c in extracted_constraints]

    # Add line overload constraint
    net['line']['max_loading_percent'] = 100
    extracted_constraints = constraints.create_default_constraints(net, {})
    assert len(extracted_constraints) == 2
    assert constraints.LineOverloadConstraint in [type(c) for c in extracted_constraints]

    # Add trafo overload constraint
    net['trafo']['max_loading_percent'] = 100
    extracted_constraints = constraints.create_default_constraints(net, {})
    assert len(extracted_constraints) == 3
    assert constraints.TrafoOverloadConstraint in [type(c) for c in extracted_constraints]

    # Add ext grid active power constraint
    net['ext_grid']['min_p_mw'] = 0
    extracted_constraints = constraints.create_default_constraints(net, {})
    assert len(extracted_constraints) == 4
    assert constraints.ExtGridActivePowerConstraint in [type(c) for c in extracted_constraints]

    # Add ext grid reactive power constraint
    net['ext_grid']['min_q_mvar'] = 0
    extracted_constraints = constraints.create_default_constraints(net, {})
    assert len(extracted_constraints) == 5
    assert constraints.ExtGridReactivePowerConstraint in [type(c) for c in extracted_constraints]

    # Existing but unconstrained constraints should not be counted because 
    # cannot result in violations anyway
    net['ext_grid']['min_q_mvar'] = -np.inf
    net['ext_grid']['max_q_mvar'] = np.inf
    net['ext_grid']['min_p_mw'] = np.nan
    net['ext_grid']['max_p_mw'] = np.nan
    net['bus']['min_vm_pu'] = None
    net['bus']['max_vm_pu'] = None
    extracted_constraints = constraints.create_default_constraints(net, {})
    assert len(extracted_constraints) == 2
    assert constraints.ExtGridReactivePowerConstraint not in [type(c) for c in extracted_constraints]
    assert constraints.ExtGridActivePowerConstraint not in [type(c) for c in extracted_constraints]
    assert constraints.VoltageConstraint not in [type(c) for c in extracted_constraints]


def test_custom_constraint_def(net):
    get_values = lambda net: net.res_sgen.p_mw / 2
    get_boundaries = lambda net: {'min': 0, 'max': net.sgen.max_p_mw}

    custom_constraint = constraints.Constraint(
        'sgen', 'p_mw', get_values=get_values, get_boundaries=get_boundaries)

    net['sgen']['min_p_mw'] = 0.8  # Should be ignored (see get_boundaries)
    net['sgen']['max_p_mw'] = 1
    net.res_sgen['p_mw'] = 1.5  # Should be interpreted as 1.5/2 = 0.75
    result = custom_constraint(net)
    assert result['valid']

    net.res_sgen['p_mw'] = 3.0
    result = custom_constraint(net)
    assert not result['valid']
    assert result['violation'] == 0.5  # 3/2 - 1