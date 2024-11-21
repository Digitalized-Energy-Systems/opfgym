import pytest

import opfgym.util as util


def test_module_loading():
    from opfgym import reward
    loaded_class = util.load_class_from_module('Summation', 'opfgym.reward')
    assert isinstance(loaded_class(), reward.Summation)

    loaded_class = util.load_class_from_module('Replacement', 'opfgym.reward')
    assert isinstance(loaded_class(), reward.RewardFunction)

    from opfgym import constraints
    loaded_class = util.load_class_from_module(
        'VoltageConstraint', 'opfgym.constraints')
    assert isinstance(loaded_class(), constraints.Constraint)
