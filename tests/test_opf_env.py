import numpy as np
import pytest

from mlopf.envs import MaxRenewable
import mlopf.opf_env as opf_env


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


def test_test_share_def():
    all_steps = dummy_env.profiles[('sgen', 'p_mw')].index

    # Test deterministic dataset creation
    test_steps, validation_steps, train_steps = opf_env.define_test_train_split(test_share=0.1)
    assert test_steps[0] == 0
    assert validation_steps[0] == 672
    test_steps2, validation_steps2, train_steps = opf_env.define_test_train_split(test_share=0.1)
    assert (set(test_steps) == set(test_steps2))
    assert (set(validation_steps) == set(validation_steps2))

    # Make sure there is no overlap
    assert set(validation_steps).isdisjoint(test_steps)
    assert set(validation_steps).isdisjoint(train_steps)

    # Test stochastic dataset creation
    test_steps, validation_steps, train_steps = opf_env.define_test_train_split(test_share=0.1, random_validation_steps=True, random_test_steps=True)
    test_steps2, validation_steps2, train_steps2 = opf_env.define_test_train_split(test_share=0.1, random_validation_steps=True, random_test_steps=True)
    assert not (set(test_steps) == set(test_steps2))
    assert not (set(validation_steps) == set(validation_steps2))

    # Make sure (again) there is no overlap
    assert set(validation_steps).isdisjoint(test_steps)
    assert set(validation_steps).isdisjoint(train_steps)

    # Size of the dataset (roughly) correct?
    assert len(all_steps) / 10.5 <= len(test_steps) <= len(all_steps) / 9.5
    test_steps, validation_steps, train_steps = opf_env.define_test_train_split(test_share=0.5)
    assert len(all_steps) / 2.1 <= len(test_steps) <= len(all_steps) / 1.9

    # Edge case: All data is test data
    test_steps, validation_steps, train_steps = opf_env.define_test_train_split(test_share=1.0, validation_share=0.0)
    assert set(test_steps) == set(all_steps)

    # Edge case: No validation data
    test_steps, validation_steps, train_steps = opf_env.define_test_train_split(validation_share=0.0)
    assert len(validation_steps) == 0

    # Only 100% of the data can be used
    with pytest.raises(AssertionError):
        opf_env.define_test_train_split(test_share=0.6, validation_share=0.6)
