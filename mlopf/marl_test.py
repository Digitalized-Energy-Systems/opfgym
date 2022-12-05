""" Simple MARL test env: Multiple cooperating agents hunt some randomly moving
prey in 2D environment. """

import gym
import numpy as np
import pettingzoo
from pettingzoo.utils import wrappers
from pettingzoo.utils import to_parallel


class HuntingEnv(gym.Env):
    def __init__(self, n_agents=5, field_size=50):
        self.observation_space = gym.spaces.Box(
            low=np.zeros(2 * (n_agents + 1)), high=field_size * np.ones(2 * (n_agents + 1)))
        self.action_space = gym.spaces.Box(
            low=-np.ones(2 * n_agents), high=np.ones(2 * n_agents))

        # All agents observe all (x, y) coordinates of all agents
        self.agent_observation_mapping = [
            np.array(range(2 * (n_agents + 1)))] * n_agents
        self.agent_action_mapping = [
            np.array([a, a + 1]) for a in range(0, 2 * n_agents, 2)]
        # All agents share the same reward with index 0
        self.agent_reward_mapping = [0] * n_agents

        self.n_agents = n_agents
        self.field_size = field_size

    def reset(self):
        self.agent_positions = [np.random.rand(
            2) for _ in range(self.n_agents)]
        self.prey_position = np.random.rand(2)
        self.step_counter = 0
        return self._get_obs()

    def step(self, action):
        # Apply actions
        for a in range(self.n_agents):
            self.agent_positions[a] += action[(2 * a):(2 * a + 2)]
            self.agent_positions[a] = np.clip(self.agent_positions[a], 0, 1)

        # Random prey movement (TODO: Maybe actively move away from hunter)
        self.prey_position += np.random.rand(2) / self.field_size
        self.prey_position = np.clip(self.prey_position, 0, 1)

        # Success condition: 3 or more agents have distance <1 to prey
        counter = 0
        prey_position = self.prey_position * self.field_size
        for a in range(self.n_agents):
            agent_position = self.agent_positions[a] * self.field_size
            distance = (sum((prey_position - agent_position)**2))**0.5
            if distance < 1:
                counter += 1

            if counter >= 3:
                print(f'Success in step {self.step_counter}!')
                done = True
                reward = 5
                break
        else:
            done = False
            reward = -1
            if self.step_counter > 200:
                done = True

        obs = self._get_obs()

        self.step_counter += 1

        return obs, reward, done, {}

    def _get_obs(self):
        return np.concatenate(self.agent_positions + [self.prey_position])

    def render(self):
        pass


def HuntingEnvPZ(**hyperparams):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = HuntingEnvPZClass(**hyperparams)
    # env = wrappers.CaptureStdoutWrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)
    # env = to_parallel(env)
    return env


class HuntingEnvPZClass(pettingzoo.ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "rps_v2"}

    def __init__(self, n_agents=5, field_size=50):
        self.possible_agents = [f'hunter_{idx}' for idx in range(n_agents)]
        self.state_space = gym.spaces.Box(
            low=np.zeros(2 * (n_agents + 1)),
            high=field_size * np.ones(2 * (n_agents + 1)))
        self.observation_spaces = {a_id: self.state_space
                                   for a_id in self.possible_agents}
        self.action_spaces = {a_id: gym.spaces.Box(
            low=-np.ones(2), high=np.ones(2)) for a_id in self.possible_agents}

        # All agents observe all (x, y) coordinates of all agents
        # self.agent_observation_mapping = [
        #     np.array(range(2 * (n_agents + 1)))] * n_agents
        # self.agent_action_mapping = [
        #     np.array([a, a + 1]) for a in range(0, 2 * n_agents, 2)]
        # All agents share the same reward with index 0
        # self.agent_reward_mapping = [0] * n_agents

        self.n_agents = n_agents
        self.field_size = field_size
        self.agent_selection = 'hunter_0'

    def reset(self):
        self.agent_positions = {
            a_id: np.random.rand(2) for _ in range(self.n_agents)
            for a_id in self.possible_agents}
        self.prey_position = np.random.rand(2)
        self.step_counter = 0
        return self._get_obs()

    def step(self, action):
        # Apply actions
        for a_id in self.possible_agents:
            self.agent_positions[a_id] += action[a_id]
            self.agent_positions[a_id] = np.clip(
                self.agent_positions[a_id], 0, 1)

        # Random prey movement (TODO: Maybe actively move away from hunter)
        self.prey_position += np.random.rand(2) / self.field_size
        self.prey_position = np.clip(self.prey_position, 0, 1)

        # Success condition: 3 or more agents have distance <1 to prey
        counter = 0
        prey_position = self.prey_position * self.field_size
        for a_id in self.possible_agents:
            agent_position = self.agent_positions[a_id] * self.field_size
            distance = (sum((prey_position - agent_position)**2))**0.5
            if distance < 1:
                counter += 1

            if counter >= 3:
                print(f'Success in step {self.step_counter}!')
                done = {a_id: True for a_id in self.possible_agents}
                reward = {a_id: 5 for a_id in self.possible_agents}
                break
        else:
            done = {a_id: False for a_id in self.possible_agents}
            reward = {a_id: -1 for a_id in self.possible_agents}
            if self.step_counter > 200:
                done = done = {a_id: True for a_id in self.possible_agents}

        obs = self._get_obs()

        self.step_counter += 1

        return obs, reward, done, {}

    def _get_obs(self):
        obs = self.state()
        return {a_id: obs for a_id in self.possible_agents}

    def state(self):
        agent_pos = [self.agent_positions[a] for a in self.possible_agents]
        return np.concatenate(agent_pos + [self.prey_position])

    def render(self):
        pass
