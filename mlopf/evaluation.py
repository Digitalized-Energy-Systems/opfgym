import random
import time

import drl.experiment
import numpy as np


def main(path, test_steps=100, **kwargs):
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

    # evaluate_nstep(agent, env, test_steps)
    measure_speedup(agent, env, test_steps)


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
            if not np.all(info['valids']):
                valids[step, n] = 0

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
    path = 'market_design_paper/data/DGX/20230530_qmarket_base/2023-05-30T15.34.54.369_QMarketEnv_Ddpg1Step_0/'
    main(path)
