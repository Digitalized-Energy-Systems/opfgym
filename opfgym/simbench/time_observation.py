
import numpy as np

def get_simbench_time_observation(current_step: int, total_n_steps: int=24*4*366):
    """ Return current time in sinus/cosinus form.
    Example daytime: (0.0, 1.0) => 00:00 and (1.0, 0.0) => 06:00. Idea from
    https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

    Results in overall 6 time observations: sin/cos for day, week, year.

    Assumes 15 min step size.
    """
    # number of steps per timeframe
    dayly, weekly, yearly = (24 * 4, 7 * 24 * 4, total_n_steps)
    time_obs = []
    for timeframe in (dayly, weekly, yearly):
        timestep = current_step % timeframe
        cyclical_time = 2 * np.pi * timestep / timeframe
        time_obs.append(np.sin(cyclical_time))
        time_obs.append(np.cos(cyclical_time))

    return np.array(time_obs)
