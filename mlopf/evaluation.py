import random
import time

import drl.experiment
import numpy as np

# TODO: Add acts to eval/MAPE


def eval_experiment_folder(path, test_steps_int=50, seed=10, **kwargs):

    rewards_dict = defaultdict(list)
    acts_dict = defaultdict(list)
    valids_dict = defaultdict(list)
    violations_dict = defaultdict(list)
    names = []
    print('Get agent performances')
    run_paths = os.listdir(path)
    for idx, run_path in enumerate(run_paths):
        print('')
        full_path = path + run_path
        print(full_path)
        agent, env, name = get_agent_data(full_path)
        names.append(name)

        rewards, acts, valids, violations = get_agent_performance(
            agent, env, test_steps_int, seed)
        rewards_dict[name].append(np.array(rewards))
        acts_dict[name].append(np.array(acts))
        valids_dict[name].append(np.array(valids))
        violations_dict[name].append(np.array(violations))

    print('Get baseline performance')
    base_rewards, base_acts = get_baseline_performance(
        env, test_steps_int, seed)
    base_rewards = np.array(base_rewards)
    base_acts = np.array(base_acts)

    print('Get metrics \n')
    for name in valids_dict.keys():
        print('Experiment: ', name)
        print('On the exact same samples: (valid for everyone)')
        performance_comparison(
            name, valids_dict, rewards_dict[name], violations_dict[name],
            base_rewards, True)
        print('\nOn samples that are valid for the respective experiment: ')
        performance_comparison(
            name, valids_dict, rewards_dict[name], violations_dict[name],
            base_rewards, False)
        print('')


# TODO
def eval_one_agent(path, test_steps=100, **kwargs):

    agent, env, name = get_agent_data(path)
    # evaluate_nstep(agent, env, test_steps)
    measure_speedup(agent, env, test_steps)


def get_agent_data(path):
    from drl.util.load_agent import load_agent
    agent = load_agent(path)

    return agent, agent.env, agent.name


def get_agent_performance(agent, env, test_steps_int, seed=None, criterion='last'):
    if seed:
        random.seed(seed)
        np.random.seed(seed=seed)
    test_steps = random.sample(list(env.test_steps), test_steps_int)

    rewards = []
    acts = []
    valids = []
    violations = []
    for step in test_steps:
        obs = env.reset(step=step, test=True)
        done = False
        best_obj = -np.inf
        while not done:
            act = agent.test_act(agent.scale_obs(obs))
            obs, reward, done, info = env.step(act)

            obj = sum(env.calc_objective(env.net))
            if criterion == 'last' or (obj > best_obj and criterion == 'best'):
                # TODO: Unclear if I should use the best or the last one for eval?
                best_obj = obj
                best_act = act
                best_valid = info['valids'].all()
                best_viol = info['violations'].sum()

        rewards.append(best_obj)
        acts.append(best_act)
        valids.append(best_valid)
        violations.append(best_viol)

    return rewards, acts, valids, violations


def get_baseline_performance(env, test_steps_int, seed=None):
    if seed:
        random.seed(seed)
        np.random.seed(seed=seed)
    test_steps = random.sample(list(env.test_steps), test_steps_int)

    rewards = []
    acts = []
    for step in test_steps:
        env.reset(step=step, test=True)
        reward = env.baseline_reward()
        # TODO Attention: Don't forget the scaling here!
        act = env.get_current_actions()

        rewards.append(reward)
        acts.append(act)

    return rewards, acts


def performance_comparison(name, valids_dict: dict, rewardss: list,
                           violationss: list, baseline: list,
                           shared_mask: False):
    """ Make sure to compute error metrics like MAPE on equal basis by
    comparing only data points with equal constraint satisfaction. """

    validss = [np.logical_and.reduce(v) for v in valids_dict.values()]
    shared_valids_mask = np.logical_and.reduce(validss)

    regrets = []
    mapes = []
    mpes = []
    rmses = []
    shares = []
    mean_violations = []

    for idx in range(len(rewardss)):
        mask = shared_valids_mask if shared_mask else valids_dict[name][idx]
        errors = (baseline - rewardss[idx])[mask]
        regrets.append(np.mean(errors))
        mapes.append(np.mean(abs(errors / baseline[mask])) * 100)
        mpes.append(np.mean(errors / abs(baseline[mask])) * 100)
        rmses.append(np.mean(errors**2)**0.5)
        shares.append(np.mean(mask))
        mean_violations.append(np.mean(violationss[idx][~mask]))

    print('Mean Regret: ', round(np.mean(regrets), 4))
    print('MAPE: ', round(np.mean(mapes), 4), '%')
    print('MPE: ', round(np.mean(mpes), 4), '%')
    print('RMSE: ', round(np.mean(rmses), 4))
    print('Valid share: ', round(np.mean(shares), 4))
    print('Mean violation (of invalid samples): ',
          round(np.mean(mean_violations), 6))

    # TODO: Store result in `path`


def measure_speedup(agent, env, test_steps, path=None):
    """ Compare computation times of conventional OPF with RL-OPF. """
    test_steps = random.sample(list(env.test_steps), test_steps)

    # Make sure that env is resetted with valid actions (for OPF)
    env.add_act_obs = True

    print('Time measurement for the conventional OPF')
    start_time = time.time()
    for n in test_steps:
        env.reset(step=n)
        env._optimal_power_flow()
    opf_time = round(time.time() - start_time, 3)

    print('Time measurement for RL')
    start_time = time.time()
    for n in test_steps:
        obs = env.reset(step=n)
        agent.test_act(obs)
    rl_time = round(time.time() - start_time, 3)
    rl_speedup = round(opf_time / rl_time, 3)

    print('Measurement for RL in batches')
    start_time = time.time()
    obss = np.concatenate([env.reset(step=n).reshape(1, -1)
                           for n in test_steps], axis=0)
    agent.test_act(obss)
    batch_time = round(time.time() - start_time, 3)
    batch_speedup = round(opf_time / batch_time, 3)

    print('Time measurement for RL as warm start for conventional OPF \n')
    start_time = time.time()
    for n in test_steps:
        obs = env.reset(step=n)
        act = agent.test_act(obs)
        env._apply_actions(act)
        env._optimal_power_flow(init='pf')
    rl_and_opftime = round(time.time() - start_time, 3)
    rl_and_opf_speedup = round(opf_time / rl_and_opftime, 3)

    if path:
        with open(path + 'time_measurement.txt', 'w') as f:
            f.write(f'Device: {agent.device} \n')
            f.write(f'Samples: {test_steps} \n')
            f.write(f'OPF time: {opf_time} \n')
            f.write(f'RL time: {rl_time} (speed-up: {rl_speedup})\n')
            f.write(
                f'Batch RL time: {batch_time} (speed-up: {batch_speedup})\n')
            f.write(
                f'Warm-start time: {rl_and_opftime} (speed-up: {rl_and_opf_speedup})\n')
    else:
        print('Device: ', agent.device)
        print(f'Samples: {test_steps} \n')
        print('OPF time: ', opf_time)
        print('RL time: ', rl_time, f'(speed-up: {batch_speedup})')
        print(f'Batch RL time: {batch_time} (speed-up: {batch_speedup})')
        print(
            f'Warm-start time: {rl_and_opftime} (speed-up: {rl_and_opf_speedup})\n')


if __name__ == '__main__':
    path = 'HPC/drlopf_experiments/data/final_experiments/20230802_qmarket_nstep/'
    eval_experiment_folder(path)
