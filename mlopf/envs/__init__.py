""" Register OPF environments to gymnasium. """

from gymnasium.envs.registration import register

from mlopf.envs.eco_dispatch import EcoDispatch
from mlopf.envs.max_renewable import MaxRenewable
from mlopf.envs.q_market import QMarket
from mlopf.envs.voltage_control import VoltageControl
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
    id='VoltageControl-v0',
    entry_point='mlopf.envs:VoltageControl',
)

register(
    id='EcoDispatch-v0',
    entry_point='mlopf.envs:EcoDispatch',
)

register(
    id='LoadShedding-v0',
    entry_point='mlopf.envs:LoadShedding',
)
