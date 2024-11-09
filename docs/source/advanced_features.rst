Advanced Features by Examples
===============================

While the :ref:`Benchmarks` focus on the standard OPF that is deterministic, 
single-time-step, and has continuous actions, *OPF-Gym* also allows for more 
advanced OPF problems.

Note that all the following examples are not solvable with the conventional 
pandapower OPF solver anymore. Therefore, either a custom solver is required or
a comparison with ground-truth solutions is not possible.

Multi-Stage OPF
---------------
The multi-stage OPF problem is an OPF that is performed over multiple time 
steps, including constraint satisfaction over multiple time steps, for example,
storage state-of-charge or ramping constraints.
The multi-stage OPF can be implemented by overwriting the :meth:`step` method, as 
shown in the 
`multi-stage OPF example <https://github.com/Digitalized-Energy-Systems/opfgym/blob/development/opfgym/examples/multi_stage.py>`_.

Security-Constrained OPF
------------------------
The security-constrained OPF problem is an OPF were all constraints are also
consideref for the N-1 case with line outages. It can be implemented by adding
a loop to the :meth:`calculate_violations` method, as shown in the
`security-constrained OPF example <https://github.com/Digitalized-Energy-Systems/opfgym/blob/development/opfgym/examples/security_constrained.py>`_.

Mixed Continuous and Discrete Actions
-------------------------------------
While conventional solvers have difficulties with discrete actions, adding
discrete actions in *OPF-Gym* is straightforward and does not complicate the 
learning problem nor the environment creation, as shown in the 
`Mixed actuators example <https://github.com/Digitalized-Energy-Systems/opfgym/blob/development/opfgym/examples/mixed_continuous_discrete.py>`_. 
Note that **all** actions in *OPF-Gym* are modelled as continuous RL actions and
require continuous RL algorithms like DDPG or SAC, even when they are discrete
from an energy system perspective.

Stochastic OPF
--------------
The OPF problem can be extended to include stochasticity, for example, when the 
measurement of the grid state (voltages, power flows) is uncertain or noisy. 
A simple example of an stochastic OPF environment is shown in the 
`Stochastic OPF example <https://github.com/Digitalized-Energy-Systems/opfgym/blob/development/opfgym/examples/stochastic_obs.py>`_.

Partial Observability
---------------------
Conventional solvers require an estimation of the full state of the power grid.
RL can also be applied to partially observable environments, where only a
subset of the state is observable. This is shown in the 
`Partial Observability example <https://github.com/Digitalized-Energy-Systems/opfgym/blob/development/opfgym/examples/partial_obs.py>`_.

