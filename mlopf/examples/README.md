# Examples
Some example environments that show the capabilities of this framework. Since 
these problems are not solvable with the pandapower OPF, they are not part of
the benchmark suite. The example environments are:

* `NetworkReconfiguration`: This example environment demonstrates how discrete 
binary switches (closed/open) can be used for network reconfiguration tasks. 
It also demonstrates how transformer tap changers can be used as actions. 
Not solvable with pandapower OPF because it does not allow for discrete actions. 
* `MultiStageOpf`: Demonstrates how to implement an OPF over multiple time steps. 
The agent now has to make long-term decisions regarding its actions and cannot 
greedily optimize the current grid state. 

To import the example environments, use `from mlopf.examples import <EnvironmentName>`.