import pytest
import simbench as sb

import opfgym.simbench.build_simbench_net as build_simbench


@pytest.fixture
def net():
    return sb.get_simbench_net('1-LV-rural1--0-sw')


def test_system_constraint_setting(net):
    build_simbench.set_system_constraints(
        net, voltage_band=0.1, max_loading=95)
    assert (net.bus.max_vm_pu == 1.1).all()
    assert (net.bus.min_vm_pu == 0.9).all()
    assert (net.line.max_loading_percent == 95).all()
    assert (net.trafo.max_loading_percent == 95).all()

    build_simbench.set_system_constraints(net, max_loading=120)
    assert (net.line.max_loading_percent == 120).all()
    assert (net.trafo.max_loading_percent == 120).all()
    # Voltage constraints should remain unchanged!
    assert (net.bus.max_vm_pu == 1.1).all()
    assert (net.bus.min_vm_pu == 0.9).all()


def test_scaling_setting(net):
    build_simbench.set_unit_scaling(net, gen_scaling=1.2, load_scaling=1.5)

    assert (net.sgen.scaling == 1.2).all()
    assert (net.gen.scaling == 1.2).all()
    assert (net.load.scaling == 1.5).all()
    assert (net.storage.scaling == 1.0).all()


@pytest.fixture
def net_profile():
    """ Create a large simbench system plus profile data. """
    net = sb.get_simbench_net('1-MV-comm--1-sw')

    assert not sb.profiles_are_missing(net)
    profiles = sb.get_absolute_values(
        net, profiles_instead_of_study_cases=True)
    return net, profiles


def test_simbench_profile_repair(net_profile):
    net, profiles = net_profile
    len_sgen = len(net.sgen.index)

    build_simbench.repair_simbench_profiles(net, profiles)

    # Some generators should be removed (otherwise test badly designed)
    assert len(net.sgen.index) < len_sgen

    assert profiles[('sgen', 'p_mw')].min().min() >= 0.0

    for type_act in profiles.keys():
        assert (profiles[type_act].max(
            axis=0) != profiles[type_act].min(axis=0)).all()


def test_unit_constraint_setting(net_profile):
    net, profiles = net_profile
    build_simbench.set_constraints_from_profiles(net, profiles)

    for unit_type in ('ext_grid', 'sgen', 'load'):
        for column in ('p_mw', 'q_mvar'):
            # Excption: Boundaries for sgen+reactive not defined
            if not (unit_type == 'sgen' and column == 'q_mvar'):
                assert f'max_max_{column}' in net[unit_type].columns
                assert f'min_min_{column}' in net[unit_type].columns
