import numpy as np
import pytest

from opfgym.envs import MaxRenewable
import opfgym.opf_env as opf_env


dummy_env = MaxRenewable()

def test_obs_space_def():
    dummy_env.reset()
    obs_keys = (
        ('sgen', 'p_mw', np.array([46])),
        ('sgen', 'q_mvar', np.array([46])),
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


def test_get_current_actions():
    # TODO: Better would be to have a deterministic test here
    for _ in range(100):
        dummy_env.reset()
        random_action = dummy_env.action_space.sample()
        dummy_env.step(random_action)
        current_action = dummy_env.get_current_actions()
        print(random_action, current_action)
        print(random_action == current_action)
        assert np.allclose(random_action, current_action)
