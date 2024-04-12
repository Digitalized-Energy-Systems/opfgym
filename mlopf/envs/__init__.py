""" Register OPF environments to gymnasium. """

from gymnasium.envs.registration import register

from mlopf.envs.thesis_envs import MaxRenewable, QMarketEnv, EcoDispatchEnv, VoltageControlEnv
from mlopf.envs.load_shedding import LoadShedding


register(
    id='MaxRenewable-v0',
    entry_point='mlopf.envs:MaxRenewable',
)

register(
    id='QMarketEnv-v0',
    entry_point='mlopf.envs:QMarketEnv',
)

register(
    id='EcoDispatchEnv-v0',
    entry_point='mlopf.envs:EcoDispatchEnv',
)
