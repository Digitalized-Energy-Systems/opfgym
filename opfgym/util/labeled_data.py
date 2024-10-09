""" Create a labeled dataset for supervised learning of a given opfgym 
environment. """

import logging
import os
from typing import Tuple

import pandas as pd
import numpy as np

from opfgym.opf_env import OpfEnv

logger = logging.getLogger(__name__)


def create_labeled_dataset(
        env: OpfEnv,
        num_samples: int,
        keep_invalid_samples: bool=False,
        store_to_path: str=None,
        seed: int=None
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Return a labeled dataset for a given environment. The dataset consists
    of inputs (observations), outputs (optimal actions), and objectives (maybe
    useful for scaling).

    Args:
        env: opfgym environment
        num_samples: number of samples to generate
        store_to_path: path to save the dataset (do not save if None)
        seed: seed for reproducibility

    Returns:
        tuple: (observations/inputs, actions/outputs, objectives)
    """

    if seed is not None:
        # TODO: This is not the best way to set the seed. Use the gymnasium seed instead. Requires change in the base class.
        np.random.seed(seed)

    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    inputs = np.zeros((num_samples, n_observations))
    outputs = np.zeros((num_samples, n_actions))
    objectives = np.zeros(num_samples)

    counter = 0
    while counter < num_samples:
        logger.info(f'Create sample {counter+1}/{num_samples}')
        obs, info = env.reset(seed=seed+counter)
        optimal_objective = env.baseline_objective()

        if np.isnan(optimal_objective):
            continue

        if not env.is_valid():
            if not keep_invalid_samples:
                logger.warning(f'Invalid state in sample {counter}. Skip sample.')
                continue
            logger.warning(f'Invalid state in sample {counter}. Please check if the OPF solver in the environment is working correctly.')

        inputs[counter] = obs
        outputs[counter] = env.get_current_actions()
        objectives[counter] = optimal_objective
        counter += 1

    if store_to_path is not None:
        os.makedirs(store_to_path, exist_ok=True)
        # TODO: Adding headers would actually be useful maybe
        pd.DataFrame(inputs).to_csv(os.path.join(store_to_path, 'inputs.csv'), index=False, header=False)
        pd.DataFrame(outputs).to_csv(os.path.join(store_to_path, 'outputs.csv'), index=False, header=False)
        pd.DataFrame(objectives).to_csv(os.path.join(store_to_path, 'optimal_objectives.csv'), index=False, header=False)

    return inputs, outputs, objectives


if __name__ == "__main__":
    print('Example run:')
    from opfgym.envs import QMarket
    logger.setLevel(logging.INFO)
    create_labeled_dataset(QMarket(), num_samples=10, store_to_path='supervised_data_qmarket/', seed=42)
