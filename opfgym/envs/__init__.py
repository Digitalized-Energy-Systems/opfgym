""" Register OPF environments to gymnasium. """

from gymnasium.envs.registration import register

from opfgym.envs.eco_dispatch import EcoDispatch
from opfgym.envs.max_renewable import MaxRenewable
from opfgym.envs.q_market import QMarket
from opfgym.envs.voltage_control import VoltageControl
from opfgym.envs.load_shedding import LoadShedding


register(
    id='MaxRenewable-v0',
    entry_point='opfgym.envs:MaxRenewable',
)

register(
    id='QMarket-v0',
    entry_point='opfgym.envs:QMarket',
)

register(
    id='VoltageControl-v0',
    entry_point='opfgym.envs:VoltageControl',
)

register(
    id='EcoDispatch-v0',
    entry_point='opfgym.envs:EcoDispatch',
)

register(
    id='LoadShedding-v0',
    entry_point='opfgym.envs:LoadShedding',
)
