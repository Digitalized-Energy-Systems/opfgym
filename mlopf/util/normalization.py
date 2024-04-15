
import numpy as np


def get_normalization_params(env, num_samples=1000):
    """ Get normalization parameters for scaling down the reward. """
    objectives = []
    violations = []
    for _ in range(num_samples):
        # Apply random actions to random steps
        env._sampling()
        env._apply_actions(env.action_space.sample())
        env._run_pf()
        objectives.append(env.calc_objective(env.net))
        # TODO: These are the penalties, not the violations currently!
        violations.append(env.calc_violations()[2])

    norm_params = {
        'min_obj': np.array(objectives).sum(axis=1).min(),
        'max_obj': np.array(objectives).sum(axis=1).max(),
        'min_viol': np.array(violations).sum(axis=1).min(),
        'max_viol': np.array(violations).sum(axis=1).max(),
        'mean_obj': np.sum(objectives, axis=1).mean(),
        'mean_viol': np.sum(violations, axis=1).mean(),
        'std_obj': np.std(np.sum(objectives, axis=1)),
        'std_viol': np.std(np.sum(violations, axis=1)),
        'median_obj': np.median(np.sum(objectives, axis=1)),
        'median_viol': np.median(np.sum(violations, axis=1)),
        'mean_abs_obj': np.abs(np.sum(objectives, axis=1)).mean(),
        'mean_abs_viol': np.abs(np.sum(violations, axis=1)).mean(),
        # Calculate lowest 5% of the objectives to potentially clip outliers
        'low5_percentil_obj': np.percentile(np.sum(objectives, axis=1), 5),
        'low5_percentil_viol': np.percentile(np.sum(violations, axis=1), 5),
        'top5_percentil_obj': np.percentile(np.sum(objectives, axis=1), 95),
        'top5_percentil_viol': np.percentile(np.sum(violations, axis=1), 95),
    }

    print(f'Normalization parameters for {env}: {norm_params}')

    return norm_params


if __name__ == '__main__':
    default_params = {'reward_factor': 1, 'reward_bias': 0, 
                      'penalty_factor': 1, 'penalty_bias': 0}
    
    print('Running normalization for QMarketeEnv')
    from mlopf.envs import QMarket
    env = QMarket(normalization_params_=default_params)
    get_normalization_params(env, 1000)

    print('Running normalization for EcoDispatchEnv')
    from mlopf.envs import EcoDispatch
    env = EcoDispatch(normalization_params_=default_params)
    get_normalization_params(env, 1000)
