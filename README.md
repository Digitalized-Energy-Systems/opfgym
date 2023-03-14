### General
A set of benchmark environments to solve the Optimal Power Flow (OPF) problem
with reinforcement learning (RL) algorithms. All environments use the openai
gym interface. (exception: `env.render()` not implemented)

### Installation
Clone the repository and run `pip install -e .` within some kind of virtual env

### Environments
Currently, three OPF environments with the openai gym environment are available:

####Standard OPF (SimpleOpfEnv)
Run `from mlopf.envs.thesis_envs import SimpleOpfEnv` to import this env.
This env is the simplest one to learn. The objective is to maximize renewable
generation subject to constraints.

#### Reactive power market (QMarketEnv)
Run `from mlopf.envs.thesis_envs import QMarketEnv` to import this env.
This env had intermediate difficulty. The objective is the minimize costs and
reactive power costs in a local reactive power market.

#### Economic dispatch (EcoDispatchEnv)
Run `from mlopf.envs.thesis_envs import EcoDispatchEnv` to import this env.
This is the most difficult environment. The goal is to perform an economic
dispatch, i.e. to minimize active power costs subject to constraints.

#### Multi-agent bidding in reactive power market (BiddingQMarketEnv)
TODO

### OPF parameters
All OPF environments are customizable. Parameters are:
* `simbench_network_name`: Define which simbench system to use (see table)
* `gen_scaling`: Define how much to upscale the generators (e.g. to create more potential constraint violations and therefore more difficult problems)
* `load_scaling`: Equivalent to `gen_scaling`
* `voltage_band`: Define the voltage band (default `0.05` for +-0.05pu)
* `max_loading`: Define the maximum load of lines and trafos (default `80` for 80%)

### Grid Notes
For every environment, different simbench/pandapower energy systems can be
choosen. The difficulty of the learning problem depends mainly on the number of
generators (~number of actuators) and the number of buses (~number of sensors
and ~complexity of the underlying function).

To decide which system to use for experiments, here a quick list with the
relevant information for each simbench system for quick access:
(Insert 0,1,2 for current, future and far future system)

| simbench_network_name   | n_buses   | n_ext_grid    | n_gen     | n_sgen        | n_loads   |
|---|---|---|---|---|---|
| 1-EHV-mixed--<0,1,2>-sw | 3085      | 7             | 338       | 225/233/241 *(225/225/225)   | 390       |
| 1-HV-mixed--<0,1,2>-sw  | 306/355/377       | 3             | 0         | 103/109/124 *(57/63/78) | 58        |
| 1-HV-urban--<0,1,2>-sw  | 372/402/428       | 1             | 0         | 98/101/118 *(42/45/62)  | 79        |
| 1-MV-rural--<0,1,2>-sw  | 97/99/99        | 1             | 0         | 102       | 96        |
| 1-MV-semiurb--<0,1,2>-sw| 117/122/122       | 1             | 0         | 121/123/123       | 115/118/122       |
| 1-MV-urban--<0,1,2>-sw  | 144       | 1             | 0         | 134       | 139       |
| 1-MV-comm--<0,1,2>-sw   | 107/110/111       | 1             | 0         | 89/90/90 *(89/89/89)       | 98/98/106        |
| 1-LV-rural1--<0,1,2>-sw | 15        | 1             | 0         | 4/8/8         | 13/14/28        |
| 1-LV-rural2--<0,1,2>-sw | 97        | 1             | 0         | 8/9/11         | 99/103/118        |
| 1-LV-rural3--<0,1,2>-sw | 129       | 1             | 0         | 17/25/27  | 118/145/153 |
| 1-LV-semiurb4--<0,1,2>-sw| 44       | 1             | 0         | 1/1/6         | 41/44/58        |
| 1-LV-semiurb5--<0,1,2>-sw | 111     | 1             | 0         | 9/14/15         | 104/118/129       |
| 1-LV-urban6--<0,1,2>-sw | 59        | 1             | 0         | 5/7/12         | 111/112/135       |

Asterisk: Generators with non-zero active power (reason for zero power unknown).
They are automatically removed from the system.

Attention: All constraints and other variables are tuned for the default
simbench systems. Whenever, you change the simbench system, it could happen
that the OPF is not solvable anymore, e.g. because the constraints are too tight.


### How to create a new env?
TODO: What needs to be done if you want to implement your own OPF environment? (action_space, observation_space, sampling, etc)
