
import pytest

from opfgym.util.normalization import get_normalization_params
from opfgym.envs import MaxRenewable


def test_get_normalization_params():
    env = MaxRenewable()
    norm_params = get_normalization_params(env, num_samples=5)
    assert isinstance(norm_params, dict)
    assert 'min_obj' in norm_params
    assert isinstance(norm_params['min_obj'], (int, float))
    assert norm_params['min_obj'] < norm_params['max_obj']
