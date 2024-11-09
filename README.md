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
planned for winter 2024. Afterward, the benchmarks will be kept as 
stable as possible. 

If you want to use the benchmark environments or the general framework to build 
own environments, please cite the following publication, where the framework is 
first mentioned (in an early stage): https://doi.org/10.1016/j.egyai.2024.100410


### Installation
Run `pip install opfgym` within some kind of virtual env.
For contributing, clone the repository and run `pip install -e .`.
Tested for python 3.10.


### Environments
Currently, five OPF benchmark environments are available. 

* EcoDispatch: Economic dispatch
* VoltageControl: Voltage Control with reactive power
* MaxRenewable: Maximize renewable feed-in
* QMarket: Reactive power market
* LoadShedding: Load shedding problem

Additionally, some 
example environments for more advanced features can be found in `opfgym/examples`. 

### Working With the Framework
All environments use the gymnasium API:
* Use `env.reset()` to start a new episode ([see gymnasium docs](https://gymnasium.farama.org/index.html))
* Use `env.step(action)` to apply an action to the environment ([see gymnasium docs](https://gymnasium.farama.org/index.html))
* Use `env.render()` to render the underlying power grid. For documentation of the usable keyword arguments, refer to the [pandapower documentation](https://pandapower.readthedocs.io/en/latest/plotting/matplotlib/simple_plot.html): 

On top, some additional OPF-specfic features are implemented: 
* Use `env.run_optimal_power_flow` to run an OPF on the current state. Returns True if successful, False otherwise. 
* Use `env.get_optimal_objective()` to return the optimal value of the objective function. Warning: Run `env.run_optimal_power_flow()` beforehand!
* Use `sum(env.calculate_objective())` to compute the value of the objective function in the current state. (Remove the `sum()` to get a vector representation)
* Use `env.get_current_actions()` to get the currently applied actions (e.g. generator setpoints). Warning: The actions are always scaled to range [0, 1] and not directly interpretable as power setpoints! 0 represents the minimum
possible setpoint, while 1 represents the maximum setpoint. 
* `env.is_state_valid()` to check if the current power grid state contains any 
constraint violations. 
* `env.is_optimal_state_valid()` to check if the power grid state contains any 
constraint violations after running the OPF. 
* Work-in-progress (TODO: `env.get_current_setpoints()`, `error_metrics` etc.)

### Contribution
Any kind of contribution is welcome! Feel free to create issues or merge 
requests. Also, additional benchmark environment are highly appreciated. For 
example, the `examples` environments could be refined to difficult but solvable
RL-OPF benchmarks. Here, it would be especially helpful to incorporate an OPF
solver that is more capable than the very limited pandapower OPF. For example, 
it should be able to deal with multi-stage problems, discrete actuators like
switches, and stochastic problems, which the pandapower OPF can't. 
For questions, feedback, collaboration, etc., contact thomas.wolgast@uni-oldenburg.de.
