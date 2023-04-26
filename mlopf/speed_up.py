
import time

import numpy as np


def measure_speedup(env, agent, samples=100, path=None):
    start_time = time.time()
    for n in range(samples):
        env.reset(step=n)
        env.baseline_reward()
    opf_time = round(time.time() - start_time, 3)

    start_time = time.time()
    for n in range(samples):
        obs = env.reset(step=n)
        agent.test_act(obs)
    rl_time = round(time.time() - start_time, 3)
    rl_speedup = round(opf_time / rl_time, 3)

    # Measure with batches
    start_time = time.time()
    obss = np.concatenate([env.reset(step=n).reshape(1, -1)
                           for n in range(samples)], axis=0)
    agent.test_act(obss)
    batch_time = round(time.time() - start_time, 3)
    batch_speedup = round(opf_time / batch_time, 3)

    if path:
        with open(path + 'time_measurement.txt', 'w') as f:
            f.write(f'Device: {agent.device} \n')
            f.write(f'Samples: {samples} \n')
            f.write(f'OPF time: {opf_time} \n')
            f.write(f'RL time: {rl_time} (speed-up: {rl_speedup})\n')
            f.write(f'Batch time: {batch_time} (speed-up: {batch_speedup})\n')
    else:
        print('Device: ', agent.device)
        print(f'Samples: {samples} \n')
        print('OPF time: ', opf_time)
        print('RL time: ', rl_time, f'(speed-up: {batch_speedup})')
        print(f'Batch time: {batch_time} (speed-up: {batch_speedup})')
