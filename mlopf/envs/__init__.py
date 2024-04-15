""" Register OPF environments to gymnasium. """

from gymnasium.envs.registration import register

from mlopf.envs.thesis_envs import MaxRenewable, QMarket, EcoDispatch, VoltageControl
from mlopf.envs.load_shedding import LoadShedding


register(
    id='MaxRenewable-v0',
    entry_point='mlopf.envs:MaxRenewable',
)

register(
    id='QMarket-v0',
    entry_point='mlopf.envs:QMarket',
)

register(
    id='EcoDispatch-v0',
    entry_point='mlopf.envs:EcoDispatch',
)
