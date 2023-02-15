""" Register OPF environments to openai gym. """

from gym.envs.registration import register

register(
    id='SimpleOpfEnv-v0',
    entry_point='mlopf.envs.thesis_envs:SimpleOpfEnv',
)

register(
    id='QMarketEnv-v0',
    entry_point='mlopf.envs.thesis_envs:QMarketEnv',
)

register(
    id='EcoDispatchEnv-v0',
    entry_point='mlopf.envs.thesis_envs:EcoDispatchEnv',
)
