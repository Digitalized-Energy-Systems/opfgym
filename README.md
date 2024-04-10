### General
A set of benchmark environments to solve the Optimal Power Flow (OPF) problem
with reinforcement learning (RL) algorithms. All environments use the gymnasium 
interface. (exception: `env.render()` not implemented)

Warning: The whole repository is work in progress. Feel free to use the 
environments as benchmarks for your research. However, the environments can be 
expected to change slightly in the next months. The release of version 1.0 is 
planned for about fall/winter 2024.   

### Installation
Clone the repository and run `pip install -e .` within some kind of virtual env.
Tested for python 3.8 (newer version will probably not work).

### Environments
Currently, three OPF environments are available:

#### Simple OPF (SimpleOpfEnv)
Use `from mlopf.envs import SimpleOpfEnv` to import this env.
The observation space has 253 dimensions and the action space 34 dimensions.
This env is the simplest one to learn. The objective is to maximize renewable
generation subject to constraints.

#### Woltage control (VoltageControlEnv)
Use `from mlopf.envs import VoltageControlEnv` to import this env.
The observation space has 315 dimensions and the action space 43 dimensions.
The goal is to find optimal reactive power setpoints to minimize losses in the 
system subject to constraints (mainly voltage level).
This env has intermediate difficulty and focuses on constraint satisfaction.

#### Reactive power market (QMarketEnv)
Use `from mlopf.envs import QMarketEnv` to import this env.
The observation space has 233 dimensions and the action space 5 dimensions.
The reactive power market is an extension of the voltage control problem. 
The objective is the minimize losses and reactive power costs in a local 
reactive power market subject to constraints. 
This env is more difficult regarding optimization because the agent has to find
the cheapest reactive power sources on the market to achieve the same goal. 

#### Economic dispatch (EcoDispatchEnv)
Use `from mlopf.envs import EcoDispatchEnv` to import this env.
The observation space has 201 dimensions and the action space 42 dimensions.
This is the most difficult environment. The goal is to perform an economic
dispatch, i.e. to minimize active power costs subject to constraints.

#### Load shedding LoadShedding
Use `from mlopf.envs import LoadShedding` to import this env. 
The observation space has 163 dimensions and the action space 55 dimensions.
The goal is to perform cost-minimal load shedding and demand response subject 
to constraints. Difficulty not tested yet (TODO).

### OPF parameters
All OPF environments are customizable. Parameters are:
* `simbench_network_name`: Define which simbench system to use (see table)
* `gen_scaling`: Define how much to upscale the generators (e.g. to create more potential constraint violations and therefore more difficult problems)
* `load_scaling`: Equivalent to `gen_scaling`
* `voltage_band`: Define the voltage band (default `0.05` for +-0.05pu)
* `max_loading`: Define the maximum load of lines and trafos (default `80` for 80%)
* Work in progress (TODO)

However, if you want your results to be comparable with other research, you 
should stick to the default settings. 

### Simbench energy systems
For every environment, different simbench/pandapower energy systems can be
choosen. The difficulty of the learning problem depends mainly on the number of
generators (~number of actuators) and the number of buses (~number of sensors
and ~complexity of the underlying function).

To decide which system to use for experiments, here a quick list with the
relevant information for each simbench system for quick access:
(Insert 0,1,2 for current, future and far future system, see simbench documentation)

| simbench_network_name   | n_buses   | n_ext_grid    | n_gen     | n_sgen        | n_loads   | n_storage   |
|---|---|---|---|---|---|---|
| 1-EHV-mixed--<0,1,2>-sw | 3085      | 7             | 338       | 225/233/241 *(225/225/225)   | 390       | 0/4/5 |
| 1-HV-mixed--<0,1,2>-sw  | 306/355/377       | 3             | 0         | 103/109/124 *(57/63/78) | 58        | 0/12/17 |
| 1-HV-urban--<0,1,2>-sw  | 372/402/428       | 1             | 0         | 98/101/118 *(42/45/62)  | 79        | 0/13/16 |
| 1-MV-rural--<0,1,2>-sw  | 97/99/99        | 1             | 0         | 102       | 96        | 0/53/90 |
| 1-MV-semiurb--<0,1,2>-sw| 117/122/122       | 1             | 0         | 121/123/123       | 115/118/122       | 0/87/114 |
| 1-MV-urban--<0,1,2>-sw  | 144       | 1             | 0         | 134       | 139       | 0/101/133 |
| 1-MV-comm--<0,1,2>-sw   | 107/110/111       | 1             | 0         | 89/90/90 *(89/89/89)       | 98/98/106        | 0/52/80 |
| 1-LV-rural1--<0,1,2>-sw | 15        | 1             | 0         | 4/8/8         | 13/14/28        | 0/4/5 |
| 1-LV-rural2--<0,1,2>-sw | 97        | 1             | 0         | 8/9/11         | 99/103/118        | 0/0/8 |
| 1-LV-rural3--<0,1,2>-sw | 129       | 1             | 0         | 17/25/27  | 118/145/153 | 0/14/16 |
| 1-LV-semiurb4--<0,1,2>-sw| 44       | 1             | 0         | 1/1/6         | 41/44/58        | 0/1/4 |
| 1-LV-semiurb5--<0,1,2>-sw | 111     | 1             | 0         | 9/14/15         | 104/118/129       | 0/10/15 |
| 1-LV-urban6--<0,1,2>-sw | 59        | 1             | 0         | 5/7/12         | 111/112/135       | 0/0/7 |

Asterisk: Generators with non-zero active power. It is unknown why some generators exist with only zero power.
They are automatically removed from the system.

Attention: All constraints and other variables are tuned for the default
simbench systems. Whenever, you change the simbench system, it could happen
that the OPF is not solvable anymore, e.g. because the constraints are too tight.


### How to create a new env?
TODO: What needs to be done if you want to implement your own OPF environment? (action_space, observation_space, sampling, etc)
