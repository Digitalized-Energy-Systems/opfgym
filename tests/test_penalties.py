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
    valid, violation, percentage_violation, penalty = penalties.voltage_violation(
        net)
    assert not valid
    assert penalty >= 0
    assert violation >= 0


def test_line_overloading(net):
    valid, violation, percentage_violation, penalty = penalties.line_overload(
        net)
    assert not valid
    assert penalty >= 0
    assert violation >= 0


def test_trafo_overloading(net):
    valid, violation, percentage_violation, penalty = penalties.trafo_overload(
        net)
    assert not valid
    assert penalty >= 0
    assert violation >= 0


def test_ext_grid_overpower(net):
    valid, violation, percentage_violation, penalty = penalties.ext_grid_overpower(
        net, column='q_mvar')
    assert not valid
    assert penalty >= 0
    assert violation >= 0


def test_compute_penalty():
    violation = 10
    n_violations = 2
    penalty = penalties.compute_penalty(
        violation, n_violations, linear_penalty=3)
    assert penalty == -30

    penalty = penalties.compute_penalty(
        violation, n_violations, offset_penalty=1.5)
    assert penalty == -3

    penalty = penalties.compute_penalty(
        violation, n_violations, offset_penalty=1.5, linear_penalty=2)
    assert penalty == -23

def test_compute_violation():
    pass # TODO
