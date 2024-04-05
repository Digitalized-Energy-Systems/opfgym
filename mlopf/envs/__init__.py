""" Register OPF environments to openai gym. """

from gym.envs.registration import register

from mlopf.envs.thesis_envs import SimpleOpfEnv, QMarketEnv, EcoDispatchEnv, VoltageControlEnv
from mlopf.envs.load_shedding import LoadShedding


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
