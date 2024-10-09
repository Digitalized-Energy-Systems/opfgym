### General
A set of benchmark environments to solve the Optimal Power Flow (OPF) problem
with reinforcement learning (RL) algorithms. It is also easily possible to create custom OPF environments. 
All environments use the [gymnasium API](https://gymnasium.farama.org/index.html). 
The modelling of the power systems and the calculation of power flows happens with
[pandapower](https://pandapower.readthedocs.io/en/latest/index.html).
The benchmark power grids and time-series data of loads and generators are 
taken from [SimBench](https://simbench.de/en/).

Warning: The whole repository is work-in-progress. Feel free to use the 
environments as benchmarks for your research. However, the environments can be 
expected to change slightly in the next months. The release of version 1.0 is 
planned for about winter 2024. Afterward, the benchmarks will be kept as 
stable as possible. 

If you want to use the benchmark environments or the general framework to build 
own environments, please cite the following publication, where the framework is 
first mentioned (in an early stage): https://doi.org/10.1016/j.egyai.2024.100410


### Installation
Clone the repository and run `pip install -e .` within some kind of virtual env.
Tested for python 3.10.


### Environments
Currently, five OPF benchmark environments are available. Additionally, some 
example environments for more advanced features can be found in `opfgym/examples`. 

#### Maximize Renewable Feed-In (MaxRenewable)
Use `from opfgym.envs import MaxRenewable` to import this env.
The observation space has 172 dimensions and the action space 18 dimensions.
This env is the simplest one to learn. The objective is to maximize renewable
generation without violating the constraints.

#### Woltage Control (VoltageControl)
Use `from opfgym.envs import VoltageControl` to import this env.
The observation space has 442 dimensions and the action space 14 dimensions.
The goal is to find optimal reactive power setpoints to minimize losses in the 
system subject to constraints (mainly voltage level). Early version first published in 
https://doi.org/10.1016/j.egyai.2024.100410.

#### Reactive Power Market (QMarket)
Use `from opfgym.envs import QMarket` to import this env.
The observation space has 305 dimensions and the action space 10 dimensions.
The reactive power market is an extension of the voltage control problem. 
The objective is to minimize loss costs and reactive power costs in a local 
reactive power market subject to constraints. 

#### Economic Dispatch (EcoDispatch)
Use `from opfgym.envs import EcoDispatch` to import this env.
The observation space has 201 dimensions and the action space 42 dimensions.
The goal is to perform an economic dispatch, i.e., to minimize active power 
costs subject to constraints. Early version first published in 
https://doi.org/10.1016/j.egyai.2024.100410.

#### Load Shedding (LoadShedding)
Use `from opfgym.envs import LoadShedding` to import this env. 
The observation space has 386 dimensions and the action space 16 dimensions.
The goal is to perform cost-minimal load shedding subject to constraints.

### Working With the Framework
All environments use the gymnasium API:
* Use `env.reset()` to start a new episode ([see gymnasium docs](https://gymnasium.farama.org/index.html))
* Use `env.step(action)` to apply an action to the environment ([see gymnasium docs](https://gymnasium.farama.org/index.html))
* Use `env.render()` to render the underlying power grid. For documentation of the usable keyword arguments, refer to the [pandapower documentation](https://pandapower.readthedocs.io/en/latest/plotting/matplotlib/simple_plot.html): 

On top, some additional OPF-specfic features are implemented: 
* Use `env.baseline_objective()` to solve the OPF with a conventional OPF solver. Returns the optimal value of the objective function. Warning: Changes the state of the power system to the optimal state!
* Use `sum(env.calc_objective())` to compute the value of the objective function in the current state. (Remove the `sum()` to get a vector representation)
* Use `env.get_current_actions()` to get the currently applied actions (e.g. generator setpoints). Warning: The actions are always scaled to range [0, 1] and not directly interpretable as power setpoints! 0 represents the minimum
possible setpoint, while 1 represents the maximum setpoint. 
* `env.is_valid()` to check if the current power grid state contains any 
constraint violations. 
* Work-in-progress (TODO: `env.get_current_setpoints()`, `error_metrics` etc.)


### Minimal Code Example
Loads one benchmark environment, performs a random action on that
environment,  computes the resulting percentage error relative to the
optimal action, and prints both actions (the optimal and suboptimal one).
Repeat three times. 
~~~
from opfgym.envs import QMarket
env = QMarket()
for _ in range(3):
    observation, info = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated: 
        # Perform random action (replace with learning agent)
        action = env.action_space.sample()  
        observation, reward, terminated, truncated, info = env.step(action)

        # Check for constraint satisfaction
        valid = info['valids'].all()
        print(f"The grid satisfies all constraints: {valid}")

        # Compute the error
        objective = sum(env.calc_objective())
        optimal_objective = env.baseline_objective()
        optimal_actions = env.get_current_actions()
        percentage_error = optimal_objective - objective / abs(optimal_objective) * 100
        print(f"Percentage error of the random action: {round(percentage_error, 2)}%")
        print(f"Optimal actions: {optimal_actions[:3]} (first three entries)")
        print(f"Agent actions: {action[:3]} (first three entries)")
        print("-------------------------------------")
~~~


### OPF Parameters
All OPF environments are customizable (with `env = QMarket(**kwargs)`). 
The parameters can be classified into two 
categories, depending on whether they change the underlying OPF problem or only 
the environment representation of the problem (e.g. the observation space).
The following parameters change the OPF problems and should only be changed if 
it is okay to change the benchmark problem: (if no comparability with other works is required)
* `simbench_network_name`: Define which simbench system to use (see table)
* `gen_scaling`: Define how much to upscale the generators (e.g. to create more potential constraint violations and therefore more difficult problems)
* `load_scaling`: Equivalent to `gen_scaling`
* `voltage_band`: Define the voltage band (default `0.05` for +-0.05pu)
* `max_loading`: Define the maximum load of lines and trafos (default `80` for 80%)
* Work-in-progress

The following parameters only change the problem representation and can be 
changed for RL environment design:
* Work-in-progress  


### Simbench Benchmark Systems
For every environment, different simbench/pandapower energy systems can be
choosen. Warning: This changes the OPF problem and makes comparability with 
other works impossible. Do this only to create new separate environments. 

The difficulty of the learning problem depends mainly on the number of
generators (~number of actuators) and the number of buses (~number of sensors
and ~complexity of the underlying function). To decide which system to use for experiments, here a quick list with the
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


### How to create a new environment?
Work-in-progress: Please check how the benchmark and examples environments 
are defined (`opfgym/envs/` and `opfgym/examples/`). 

TODO: What needs to be done if you want to implement your own OPF environment? (action_space, observation_space, sampling, etc)


### Supervised Learning Support
While the provided framework and the benchmarks are mainly intended for 
reinforcement learning, supervised learning is also possible. This way, it is 
possible to compare supervised and reinforcement learning methods on the exact
same OPF problem. To create a dataset for supervised learning, run:
~~~
from opfgym.util.labeled_data import create_labeled_dataset
from opfgym.envs import SomeEnvironment

create_labeled_dataset(SomeEnvironment(), num_samples=10, seed=42)
~~~
The dataset can be used to create an neural network in supervised fashion. 
Assuming no data scaling, the outputs can directly be fed back to `env.step()`.


### Contribution
Any kind of contribution is welcome! Feel free to create issues or merge 
requests. Also, additional benchmark environment are highly appreciated. For 
example, the `examples` environments could be refined to difficult but solvable
RL-OPF benchmarks. Here, it would be especially helpful to incorporate an OPF
solver that is more capable than the very limited pandapower OPF. For example, 
it should be able to deal with multi-stage problems, discrete actuators like
switches, and stochastic problems, which the pandapower OPF can't. 
For questions, feedback, collaboration, etc., contact thomas.wolgast@uni-oldenburg.de.
