
[PyPi](https://pypi.org/project/opfgym/) 
| [Read the Docs](https://opf-gym.readthedocs.io)
| [Github](https://github.com/Digitalized-Energy-Systems/opfgym) 
| [mail](mailto:thomas.wolgast@uni-oldenburg.de)

![lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Digitalized-Energy-Systems/opfgym/blob/development/LICENSE)
[![Test OPF-gym](https://github.com/Digitalized-Energy-Systems/opfgym/actions/workflows/test-opfgym.yml/badge.svg)](https://github.com/Digitalized-Energy-Systems/opfgym/actions/workflows/test-opfgym.yml)

### General
A set of benchmark environments to solve the Optimal Power Flow (OPF) problem
with reinforcement learning (RL) algorithms. It is also easily possible to create custom OPF environments. 
All environments use the [gymnasium API](https://gymnasium.farama.org/index.html). 
The modelling of the power systems and the calculation of power flows happens with
[pandapower](https://pandapower.readthedocs.io/en/latest/index.html).
The benchmark power grids and time-series data of loads and generators are 
taken from [SimBench](https://simbench.de/en/).

Documentation can be found on https://opf-gym.readthedocs.io/en/latest/.

Warning: The whole repository is work-in-progress. Feel free to use the 
environments as benchmarks for your research. However, the environments can be 
expected to change slightly in the next months. The release of version 1.0 is 
planned for winter 2024. Afterward, the benchmarks will be kept as 
stable as possible. 

If you want to use the benchmark environments or the general framework to build 
own environments, please cite this repository (see CITATION.cff) and/or cite
the following publication, where the framework is 
first mentioned (in an early stage): https://doi.org/10.1016/j.egyai.2024.100410



### Environments
Currently, five OPF benchmark environments are available. 

* EcoDispatch: Economic dispatch
* VoltageControl: Voltage Control with reactive power
* MaxRenewable: Maximize renewable feed-in
* QMarket: Reactive power market
* LoadShedding: Load shedding problem

Additionally, some 
example environments for more advanced features can be found in `opfgym/examples`. 

### Contribution
Any kind of contribution is welcome! Feel free to create issues or merge 
requests. Also, additional benchmark environment are highly appreciated. For 
example, the `examples` environments could be refined to difficult but solvable
RL-OPF benchmarks. Here, it would be especially helpful to incorporate an OPF
solver that is more capable than the very limited pandapower OPF. For example, 
it should be able to deal with multi-stage problems, discrete actuators like
switches, and stochastic problems, which the pandapower OPF cannot. 
For questions, feedback, collaboration, etc., contact thomas.wolgast@uni-oldenburg.de.
