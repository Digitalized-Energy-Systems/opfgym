import numpy as np
import pytest


from mlopf.envs import MaxRenewable
import mlopf.opf_env as opf_env


dummy_env = MaxRenewable()

def test_obs_space_def():
    dummy_env.reset()
    obs_keys = (
        ('sgen', 'p_mw', np.array([0])),
        ('sgen', 'q_mvar', np.array([0])),
        ('load', 'q_mvar', np.array([0])),
        ('load', 'p_mw', np.array([0])),
        ('res_bus', 'vm_pu', np.array([0])),
        ('res_line', 'loading_percent', np.array([0])),
        ('res_trafo', 'loading_percent', np.array([0])),
        ('res_ext_grid', 'p_mw', np.array([0])),
        ('res_ext_grid', 'q_mvar', np.array([0])),
    )

    obs_space = opf_env.get_obs_space(
        dummy_env.net, obs_keys, add_time_obs=False, seed=42)
    assert len(obs_space.low) == 9

    obs_space = opf_env.get_obs_space(
        dummy_env.net, obs_keys, add_time_obs=True, seed=42)
    assert len(obs_space.high) == 15

    assert not np.isnan(obs_space.low).any()
    assert not np.isnan(obs_space.high).any()


def test_action_space_def():
    act_keys = (
        ('sgen', 'p_mw', np.array([0])),
        ('sgen', 'q_mvar', np.array([0])),
        ('storage', 'p_mw', np.array([0])),
        ('gen', 'p_mw', np.array([0])),
    )

    act_space = opf_env.get_action_space(act_keys, seed=42)
    low = np.array([0.0, -1.0, -1.0, 0.0])
    high = np.array([1.0, 1.0, 1.0, 1.0])
    assert (act_space.low == low).all()
    assert (act_space.high == high).all()


def test_test_share_def():
    all_steps = dummy_env.profiles[('sgen', 'p_mw')].index

    # Test if test data share is calculated correctly
    # Some minor deviations are perfectly fine
    test_steps = opf_env.define_test_steps(test_share=0.1)
    assert len(all_steps) / 10.5 <= len(test_steps) <= len(all_steps) / 9.5
    test_steps = opf_env.define_test_steps(test_share=0.5)
    assert len(all_steps) / 2.1 <= len(test_steps) <= len(all_steps) / 1.9

    # Edge case: All data is test data
    test_steps = opf_env.define_test_steps(test_share=1.0)
    assert (all_steps == all_steps).all()

    # Edge case: No test data -> should not be done with test_share
    with pytest.raises(AssertionError):
        opf_env.define_test_steps(test_share=0.0)
