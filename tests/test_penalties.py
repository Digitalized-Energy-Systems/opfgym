import numpy as np
import pandas as pd
import pytest

from mlopf.envs.thesis_envs import SimpleOpfEnv
import mlopf.penalties as penalties


@pytest.fixture
def net():
    """ Create a pandapower net with lots of violations. """
    dummy_env = SimpleOpfEnv()
    dummy_env.reset(step=0)
    act = dummy_env.action_space.low
    dummy_env.step(act)
    return dummy_env.net


def test_voltage_violation(net):
    info = {}
    penalty, valid = penalties.voltage_violation(net, info)
    assert not valid
    assert penalty >= 0
    assert 'violations_bus_vm_pu' in info


def test_line_overloading(net):
    info = {}
    penalty, valid = penalties.line_overload(net, info)
    assert not valid
    assert penalty >= 0
    assert 'violations_line_loading_percent' in info


def test_trafo_overloading(net):
    info = {}
    penalty, valid = penalties.trafo_overload(net, info)
    assert not valid
    assert penalty >= 0
    assert 'violations_trafo_loading_percent' in info


def test_ext_grid_overpower(net):
    info = {}
    penalty, valid = penalties.ext_grid_overpower(net, info, column='q_mvar')
    assert not valid
    assert penalty >= 0
    assert 'violations_ext_grid_q_mvar' in info
