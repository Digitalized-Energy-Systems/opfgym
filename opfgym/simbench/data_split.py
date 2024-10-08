
import numpy as np


def define_test_train_split(test_share=0.2, random_test_steps=False, 
                            validation_share=0.2, random_validation_steps=False,
                            **kwargs):
    """ Return the indices of the simbench test data points. """
    assert test_share + validation_share <= 1.0
    if random_test_steps:
        assert random_validation_steps, 'Random test data does only make sense with also random validation data'

    n_data_points = 24 * 4 * 366
    all_steps = np.arange(n_data_points)

    # Define test dataset
    if test_share == 1.0:
        # Special case: Use the full simbench data set as test set
        return all_steps, np.array([]), np.array([])
    elif test_share == 0.0:
        test_steps = np.array([])
    elif random_test_steps:
        # Randomly sample test data steps from the whole year
        test_steps = np.random.choice(all_steps, int(n_data_points * test_share))
    else:
        # Use deterministic weekly blocks to ensure that all weekdays are equally represented
        # TODO: Allow for arbitrary blocks? Like days or months?
        n_test_weeks = int(52 * test_share)
        # Sample equidistant weeks from the whole year
        test_week_idxs = np.linspace(0, 51, num=n_test_weeks, dtype=int)
        one_week = 7 * 24 * 4
        test_steps = np.concatenate(
            [np.arange(idx * one_week, (idx + 1) * one_week) for idx in test_week_idxs])

    # Define validation dataset
    remaining_steps = np.array(tuple(set(all_steps) - set(test_steps)))
    if validation_share == 1.0:
        return np.array([]), all_steps, np.array([])
    elif validation_share == 0.0:
        validation_steps = np.array([])
    elif random_validation_steps:
        validation_steps = np.random.choice(remaining_steps, int(n_data_points * validation_share))
    else:
        if random_test_steps:
            test_week_idxs = np.array([])

        n_validation_weeks = int(52 * validation_share)
        # Make sure to use only validation weeks that are not already test weeks
        remaining_week_idxs = np.array(tuple(set(np.arange(52)) - set(test_week_idxs)))
        week_pseudo_idxs = np.linspace(0, len(remaining_week_idxs)-1,
                                       num=n_validation_weeks, dtype=int)
        validation_week_idxs = remaining_week_idxs[week_pseudo_idxs]
        validation_steps = np.concatenate(
            [np.arange(idx * one_week, (idx + 1) * one_week) for idx in validation_week_idxs])

    # Use remaining steps as training steps
    train_steps = np.array(tuple(set(remaining_steps) - set(validation_steps)))

    return test_steps, validation_steps, train_steps
