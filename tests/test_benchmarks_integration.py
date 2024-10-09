""" Integration tests of the benchmark environments. """

import numpy as np

from opfgym.envs import *
from .sanity_check import env_sanity_check


def test_max_renewable_integration():
    dummy_env = MaxRenewable()
    dummy_env.reset()
    for _ in range(3):
        act = dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = dummy_env.step(act)
        dummy_env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated
    assert isinstance(info, dict)
    assert env_sanity_check(dummy_env)


def test_qmarket_integration():
    dummy_env = QMarket()
    for _ in range(3):
        dummy_env.reset()
        act = dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = dummy_env.step(act)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated
    assert isinstance(info, dict)
    assert env_sanity_check(dummy_env)


def test_voltage_control_integration():
    dummy_env = VoltageControl()
    for _ in range(3):
        dummy_env.reset()
        act = dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = dummy_env.step(act)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated
    assert isinstance(info, dict)
    assert env_sanity_check(dummy_env)


def test_eco_dispatch_integration():
    dummy_env = EcoDispatch()
    for _ in range(3):
        dummy_env.reset()
        act = dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = dummy_env.step(act)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated
    assert isinstance(info, dict)
    assert env_sanity_check(dummy_env)


def test_load_shedding_integration():
    dummy_env = LoadShedding()
    for _ in range(3):
        dummy_env.reset()
        act = dummy_env.action_space.sample()
        obs, reward, terminated, truncated, info = dummy_env.step(act)

    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert terminated
    assert isinstance(info, dict)
    assert env_sanity_check(dummy_env)
