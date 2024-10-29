# Examples
Some example environments that show the capabilities of this framework. 
The environmens serve only as examples and do not represent any useful use case.
Since not all these problems are solvable with the pandapower OPF, they are not part of
the benchmark suite.The example environments are:

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
* `MixedContinuousDiscrete`: Demonstrates how continuous and discrete actuators
in the same environment are easily possible. Also demonstrates how to overwrite
the default pandapower objective function with a custom objective function
(only advisable if OPF not solvable with pandapower).
* `SecurityConstrained`: Demonstrates how a security constrained OPF can be 
implemented as RL environment, similar to, for example 
https://ieeexplore.ieee.org/abstract/document/9275611/. 
Not only checks for constraint violations in the standard case but also for the 
N-1 case of line outages. 
* `StochasticObs`: Shows how to use the pre-implemented `StochasticObservation` 
wrapper to create a simple stochastic OPF by adding noise to the observations. 

To import the example environments, use `from opfgym.examples import <EnvironmentName>`.