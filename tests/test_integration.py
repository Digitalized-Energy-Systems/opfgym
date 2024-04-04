""" Integration tests of all the default environments. """

import numpy as np

from mlopf.envs import *


def test_simple_opf_integration():
    dummy_env = SimpleOpfEnv()
    dummy_env.reset()
    for _ in range(3):
        act = dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = dummy_env.step(act)
        dummy_env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated
    assert isinstance(info, dict)


def test_qmarket_integration():
    dummy_env = QMarketEnv()
    for _ in range(3):
        dummy_env.reset()
        act = dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = dummy_env.step(act)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated
    assert isinstance(info, dict)


def test_voltage_control_integration():
    dummy_env = VoltageControlEnv()
    for _ in range(3):
        dummy_env.reset()
        act = dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = dummy_env.step(act)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated
    assert isinstance(info, dict)


def test_eco_dispatch_integration():
    dummy_env = EcoDispatchEnv()
    for _ in range(3):
        dummy_env.reset()
        act = dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = dummy_env.step(act)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated
    assert isinstance(info, dict)
