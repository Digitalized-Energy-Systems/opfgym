# Examples
Some example environments that show the capabilities of this framework. Since 
not all these problems are solvable with the pandapower OPF, they are not part of
the benchmark suite. The example environments are:

* `NetworkReconfiguration`: This example environment demonstrates how discrete 
binary switches (closed/open) can be used for network reconfiguration tasks. 
It also demonstrates how transformer tap changers can be used as actions. 
Not solvable with pandapower OPF because it does not allow for discrete actions. 
* `MultiStageOpf`: Demonstrates how to implement an OPF over multiple time steps. 
The agent now has to make long-term decisions regarding its actions and cannot 
greedily optimize the current grid state. 
* `NonSimbenchNet`: Demonstrates how to use non-Simbench power systems by the 
example of the IEEE 30-bus case. (Simbench systems are used by all benchmark 
environments because of their accompanying time-series data)
* `PartialObs`: Demonstrates how to implement missing observation data to 
create a partially observable environment. Therefore, the environment does 
not have the Markov property. 

To import the example environments, use `from opfgym.examples import <EnvironmentName>`.