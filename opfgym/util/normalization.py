"""
Problem: We want to normalize the rewards in the environment to make them 
comparable across different environments. However, we do not have any knowledge
about the reward distributions in the environments, like the minimum, maximum,
mean, median, standard deviation, etc.

Approach: We can sample the environment with random states and action pairs to
get a distribution of the rewards. We can then use this distribution to
normalize the rewards. Not perfect, but better than no normalization at all.
"""



import numpy as np


def get_normalization_params(env, 
                             num_samples=3000, 
                             clip_obj_percentiles: tuple=None, 
                             clip_viol_percentiles: tuple=None):
    """ Get normalization parameters for scaling down the reward. """
    objectives = []
    violations = []
    for _ in range(num_samples):
        # Apply random actions to random states
        env.reset()
        # Use _apply_actions() to ensure that the action space definition is kept outside (in contrast to step())
        env._apply_actions(env.action_space.sample())
        env._run_power_flow()
        objectives.append(env.calc_objective(env.net))
        # TODO: These are the penalties, not the violations currently!
        # And probably this is the right way because we cannot consider discrete penalties when we look at violations only
        violations.append(env.calc_violations()[2])

    objectives = np.array(objectives).sum(axis=1)
    violations = np.array(violations).sum(axis=1)

    # Remove potential NaNs (due to failed power flows or similar)
    objectives = objectives[~np.isnan(objectives)]
    violations = violations[~np.isnan(violations)]

    if clip_obj_percentiles:
        bottom_clip_obj = np.percentile(objectives, clip_obj_percentiles[0])
        top_clip_obj = np.percentile(objectives, clip_obj_percentiles[1])
        objectives = np.clip(objectives, bottom_clip_obj, top_clip_obj)
    else:
        bottom_clip_obj, top_clip_obj = None, None

    if clip_viol_percentiles:
        bottom_clip_viol = np.percentile(violations, clip_viol_percentiles[0])
        top_clip_viol = np.percentile(violations, clip_viol_percentiles[1])
        violations = np.clip(violations, bottom_clip_viol, top_clip_viol)
    else:   
        bottom_clip_viol, top_clip_viol = None, None

    norm_params = {
        'min_obj': objectives.min(),
        'max_obj': objectives.max(),
        'min_viol': violations.min(),
        'max_viol': violations.max(),
        'mean_obj': objectives.mean(),
        'mean_viol': violations.mean(),
        'std_obj': np.std(objectives),
        'std_viol': np.std(violations),
        'median_obj': np.median(objectives),
        'median_viol': np.median(violations),
        'mean_abs_obj': np.abs(objectives).mean(),
        'mean_abs_viol': np.abs(violations).mean(),
        # Remember at which percentiles the clipping was applied
        'clip_range_obj': (bottom_clip_obj, top_clip_obj),
        'clip_range_viol': (bottom_clip_viol, top_clip_viol),
    }

    print(f'Normalization parameters for {env}: {norm_params}')

    return norm_params


if __name__ == '__main__':
    default_params = {'reward_factor': 1, 'reward_bias': 0,
                      'penalty_factor': 1, 'penalty_bias': 0}

    print('Running normalization for QMarketeEnv')
    from opfgym.envs import QMarket
    env = QMarket(normalization_params_=default_params)
    get_normalization_params(env, 1000)

    print('Running normalization for EcoDispatchEnv')
    from opfgym.envs import EcoDispatch
    env = EcoDispatch(normalization_params_=default_params)
    get_normalization_params(env, 1000)
