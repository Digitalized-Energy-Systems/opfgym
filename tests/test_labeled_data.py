import pytest
import numpy as np

from opfgym.envs import QMarket
from opfgym.util.labeled_data import create_labeled_dataset


def test_create_labeled_dataset():
    env = QMarket()

    # Test if the dataset has the correct size
    inputs, outputs, objectives = create_labeled_dataset(
        env, num_samples=2, seed=42)
    assert inputs.shape[0] == 2
    assert outputs.shape[0] == 2
    assert len(objectives) == 2
    assert outputs.max() <= 1
    assert outputs.min() >= 0

    # Test if the dataset is reproducible
    inputs2, outputs2, objectives2 = create_labeled_dataset(
        env, num_samples=2, seed=42)
    assert (inputs == inputs2).all()
    assert (outputs == outputs2).all()
    assert (objectives == objectives2).all()

    inputs3, outputs3, objectives3 = create_labeled_dataset(
        env, num_samples=2, seed=99999)
    assert not (inputs == inputs3).all()
    assert not (outputs == outputs3).all()
    assert not (objectives == objectives3).all()
