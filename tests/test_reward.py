
import pytest

import numpy as np

import opfgym.reward as reward


def test_reward_class():
    # Test reward clipping method
    reward_fct = reward.Summation(clip_range=(0.0, 1.0))
    assert reward_fct.clip_reward(reward=1.5) == 1.0
    assert reward_fct.clip_reward(reward=-1.5) == 0.0

    # Test penalty weigth application in total reward
    reward_fct = reward.Summation(penalty_weight=0.8)
    assert reward_fct.compute_total_reward(penalty=1.0, objective=0.0) == 0.8
    assert reward_fct.compute_total_reward(penalty=0.5, objective=1.0) == 0.6
    # If none, simply add both values
    reward_fct = reward.Summation(penalty_weight=None)
    assert reward_fct.compute_total_reward(penalty=1.0, objective=0.2) == 1.2
    assert reward_fct.compute_total_reward(penalty=0.4, objective=0.5) == 0.9

    # Test reward minmax scaling in range [-1, 1]
    reward_scaling_params = {'min_objective': 2.0, 'max_objective': 10.0,
                             'min_penalty': 0.0, 'max_penalty': 5.0}
    reward_fct = reward.Summation(reward_scaling='minmax11',
                                  reward_scaling_params=reward_scaling_params)
    assert reward_fct.scale_objective(6.0) == 0.0
    assert reward_fct.scale_objective(2.0) == -1.0
    assert reward_fct.scale_objective(10.0) == 1.0
    assert reward_fct.scale_penalty(2.5) == 0.0
    assert reward_fct.scale_penalty(0.0) == -1.0
    assert reward_fct.scale_penalty(5.0) == 1.0

    # Test reward minmax scaling in range [0, 1]
    reward_scaling_params = {'min_objective': 2.0, 'max_objective': 10.0,
                             'min_penalty': 0.0, 'max_penalty': 5.0}
    reward_fct = reward.Summation(reward_scaling='minmax01',
                                  reward_scaling_params=reward_scaling_params)
    assert reward_fct.scale_objective(6.0) == 0.5
    assert reward_fct.scale_objective(2.0) == 0.0
    assert reward_fct.scale_objective(10.0) == 1.0
    assert reward_fct.scale_penalty(2.5) == 0.5
    assert reward_fct.scale_penalty(0.0) == 0.0
    assert reward_fct.scale_penalty(5.0) == 1.0

    # Test reward normalization scaling
    reward_scaling_params = {'std_objective': 2.0, 'mean_objective': 6.0,
                             'std_penalty': 1.0, 'mean_penalty': 2.5}
    reward_fct = reward.Summation(reward_scaling='normalization',
                                  reward_scaling_params=reward_scaling_params)
    assert reward_fct.scale_objective(6.0) == 0.0
    assert reward_fct.scale_objective(2.0) == -2.0
    assert reward_fct.scale_objective(8.0) == 1.0
    assert reward_fct.scale_penalty(2.5) == 0.0
    assert reward_fct.scale_penalty(1.5) == -1.0
    assert reward_fct.scale_penalty(4.5) == 2.0

def test_summation_reward():
    # Test overall reward computation (valid ignored here)
    reward_fct = reward.Summation(penalty_weight=None)
    assert reward_fct(penalty=-1.0, objective=0.0, valid=True) == -1.0
    assert reward_fct(penalty=-0.5, objective=1.0, valid=False) == 0.5
    assert reward_fct(penalty=0.0, objective=0.8, valid=True) == 0.8

def test_replacement_reward():
    reward_fct = reward.Replacement(valid_reward=0.5, penalty_weight=None)
    assert reward_fct(penalty=0.0, objective=0.2, valid=True) == 0.7
    assert reward_fct(penalty=-0.3, objective=0.2, valid=False) == -0.3
    assert reward_fct(penalty=0.0, objective=0.2, valid=False) == 0.0

def test_parameterized_reward():
    reward_fct = reward.Parameterized(valid_reward=0.7,
                                      invalid_penalty=0.3,
                                      objective_share=0.5,
                                      penalty_weight=None)
    assert reward_fct(penalty=0.0, objective=0.2, valid=True) == 0.2 + 0.7
    assert reward_fct(penalty=-0.3, objective=0.2, valid=False) == -0.3 - 0.3 + 0.2/2
