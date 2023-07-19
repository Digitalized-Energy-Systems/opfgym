
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import defaultdict
import os
import random
import time

import drl.experiment
import numpy as np

# TODO: Add acts to eval/MAPE


def eval_experiment_folder(path, test_steps_int=200, seed=10, **kwargs):

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
    for name, validss in valids_dict.items():
        performance_comparison(
            name, validss, rewards_dict[name], violations_dict[name],
            base_rewards, True)
        performance_comparison(
            name, validss, rewards_dict[name], violations_dict[name],
            base_rewards, False)
        print('')


# TODO
def eval_one_agent(path, test_steps=100, **kwargs):

    agent, env, name = get_agent_data(path)
    # evaluate_nstep(agent, env, test_steps)
    measure_speedup(agent, env, test_steps)


def get_agent_data(path):
    if path[-1] != '/':
        path += '/'

    with open(path + '/meta-data.txt') as f:
        lines = f.readlines()
    env_name = lines[1].split(' ')[1][:-1]
    algo = lines[2][15:][:-1]
    hyperparams = drl.experiment.str_to_dict(lines[6][23:][:-1])
    env_hyperparams = drl.experiment.str_to_dict(lines[7][25:][:-1])

    env = drl.experiment.create_environment(env_name, env_hyperparams)

    agent_class = drl.experiment.get_agent_class(algo)
    agent = agent_class(
        env, name='test_agent', seed=42, path=path, **hyperparams)
    agent.load_model()

    name = algo + '_' + str(hyperparams) + '_' + str(env_hyperparams)

    return agent, env, name


def get_agent_performance(agent, env, test_steps_int, seed=None):
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
        act = agent.test_act(agent.scale_obs(obs))
        obs, reward, done, info = env.step(act)

        obj = sum(env.calc_objective(env.net))
        rewards.append(obj)
        acts.append(act)
        valids.append(info['valids'].all())
        violations.append(info['violations'].sum())

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


def performance_comparison(name, validss: list, rewardss: list,
                           violationss: list, baseline: list, shared_mask: False):
    """ Make sure to compute error metrics like MAPE on equal basis by
    comparing only data points with equal constraint satisfaction. """

    shared_valids_mask = np.logical_and.reduce(validss)

    regrets = []
    mapes = []
    mpes = []
    rmses = []
    shares = []
    mean_violations = []

    print('Experiment: ', name)
    for idx in range(len(validss)):
        mask = shared_valids_mask if shared_mask else validss[idx]
        errors = (baseline - rewardss[idx])[mask]
        regrets.append(np.mean(errors))
        mapes.append(np.mean(abs(errors / baseline[mask])) * 100)
        mpes.append(np.mean(errors / abs(baseline[mask])) * 100)
        rmses.append(np.mean(errors**2)**0.5)
        shares.append(np.mean(mask))
        mean_violations.append(np.mean(violationss[idx]))

    print('Mean Regret: ', round(np.mean(regrets), 4))
    print('MAPE: ', round(np.mean(mapes), 4), '%')
    print('MPE: ', round(np.mean(mpes), 4), '%')
    print('RMSE: ', round(np.mean(rmses), 4))
    print('Valid share: ', round(np.mean(shares), 4))
    print('Mean violation: ', round(np.mean(mean_violations), 6))

    # TODO: Store result in `path`


def evaluate_nstep(agent, env, test_steps, iterations=5):
    """ Evaluate performance on n-step environment (special case!) """

    regrets = np.zeros((test_steps, iterations))
    apes = np.zeros((test_steps, iterations))
    valids = np.ones((test_steps, iterations))
    for step in range(test_steps):
        obs = agent.env.reset(test=True)
        opt_obj = agent.env.baseline_reward()
        opt_act = agent.env.get_current_actions()
        for n in range(iterations):
            act = agent.test_act(agent.scale_obs(obs))
            obs, reward, done, info = agent.env.step(act)
            obj = sum(agent.env.calc_objective(env.net))
            regrets[step, n] = opt_obj - obj
            apes[step, n] = abs(regrets[step, n] / opt_obj)
            valids[step, n] = np.all(info['valids'])

    print('mean regret: ', np.mean(regrets, axis=0))
    print('std regret: ', np.std(regrets, axis=0))
    print('MAPE: ', np.mean(apes, axis=0) * 100, '%')
    print('valid share: ', np.mean(valids, axis=0))


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
    path = 'HPC/drlopf_experiments/data/final_experiments/20230712_qmarket_baseline/'
    eval_experiment_folder(path)
