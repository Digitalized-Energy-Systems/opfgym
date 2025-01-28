Benchmarks
==========

This page lists all available RL-OPF benchmarks of *OPF-Gym*.

Ideally, the benchmarks should be loaded and used for benchmarking of new 
RL-OPF algorithms without changing the underlying OPF problem formulation.
This way, comparability of research results across multiple publications 
is ensured. 

The following table provides an overview of the benchmark environments, 
including the utilized SimBench benchmark system, the number of buses, 
number of observations, number of actions, and the implemented actuator units,
with information whether active (P) or reactive power (Q) is controlled:

+----------------+--------------------+---------+-------+----------+---------------------+
|                | Simbench System    | N Nodes | N Obs | N Action | Actuators           |
+================+====================+=========+=======+==========+=====================+
| EcoDispatch    | 1-HV-urban--0-sw   | 372     | 201   | 42       | Gens (P)            |
+----------------+--------------------+---------+-------+----------+---------------------+
| VoltageControl | 1-MV-semiurb--1-sw | 122     | 442   | 14       | Gens, Storage (Q)   |
+----------------+--------------------+---------+-------+----------+---------------------+
| QMarket        | 1-MV-rural--0-sw   | 97      | 305   | 10       | Gens (Q)            |
+----------------+--------------------+---------+-------+----------+---------------------+
| LoadShedding   | 1-MV-comm--2-sw    | 111     | 386   | 16       | Loads, Storages (P) |
+----------------+--------------------+---------+-------+----------+---------------------+
| MaxRenewables  | 1-HV-mixed--1-sw   | 355     | 172   | 18       | Gens, Storages (P)  |
+----------------+--------------------+---------+-------+----------+---------------------+

The table demonstrates that all benchmark environments utilize average sized 
power systems, resulting in difficult but solvable RL problems. Further, it
demonstrates a characteristic that is typical for RL-OPF problems, large action
and observation spaces. Both spaces are significantly larger than for typical
gymnasium environments. For example, compare with the commonly used 
`MuJoCo <https://gymnasium.farama.org/environments/mujoco/>`_ environments
that have only six actions on average.

The benchmark problems represent standard OPF problems with different 
objectives and actuators. Advanced OPF problems like multi-stage OPF or the 
security-constrained OPF are not considered (yet). However, they are possible 
in *OPF-Gym* by creating custom environments. 
See :ref:`Create Custom Environments`.

All environments share the same general constraints. They consider the voltage 
band constraints of all buses :math:`B`, 

:math:`V_{\text{min}} \leq V_b \leq V_{\text{max}} \quad \forall \; b \in B`, 

line overload constraints of all lines :math:`L`, 

:math:`S_l \leq S_l^\text{max} \quad \forall \; l \; \in \; L`,

transformer overload constraints of all transformers :math:`T`,

:math:`S_t \leq S^\text{max} \quad \forall \; t \; \in \; T`,

and the maximum active/reactive power flows over the slack bus :math:`s`,

:math:`Q^\text{min} \leq Q_s \leq Q^\text{max}`

:math:`P^\text{min} \leq P_s \leq P^\text{max}`.

Some constraints are more relevant than other for specific environments, which 
will be made explicit in the respective environment description below. 

Additionally, the minimum/maximum power setpoints of the controllable units and 
the power balance equations are considered by-design, i.e., they cannot by 
violated. 


Economic Dispatch
---------------------------------
Use :code:`from opfgym.envs import EcoDispatch` to import this environment.
The environment represents an economic dispatch problem. The goal is to 
find the optimal active power generator setpoints of a given state that 
minimize the generation costs while satisfying the slack power balance constraints.

:math:`\text{min} \; J = \sum_{a \in A} P_a \cdot p_a^P(s)`

With its 42 actions, this is the most difficult-to-solve environment.

Voltage Control
---------------------------------
Use :code:`from opfgym.envs import VoltageControl` to import this environment.
It represents a voltage control problem. The goal is to find 
the optimal reactive power setpoints of generators and a single storage system 
to minimize the overall power losses while satisfying especially the voltage band
constraints.

:math:`\text{min} \; J = P_\text{loss}`

Load Shedding
--------------------------------
Use :code:`from opfgym.envs import LoadShedding` to import this environment.
The environment represents a load shedding problem. The goal is to find the
least amount of load shedding that is required to satisfy all constraints.
All loads are assigned with different priorities, represented by their 
respective load shedding prices. Further, the agent can utilize storage systems
to reduce the amount of load shedding required.

:math:`\text{min} \; J = \sum_{a \in A} P_a \cdot p_a^P(s)`

Reactive Power Market
---------------------------------
Use :code:`from opfgym.envs import QMarket` to import this environment.
It represents a reactive power market problem, which is an 
extension to the voltage control problem. The goal is again to find optimal 
reactive power setpoints of generators that minimize the overall power losses. 
Additionally, this time, the reactive power providers need to be paid. Therefore,
a trade-off needs to be found between minimizing power losses and minimizing
the reactive power costs.

:math:`\text{min} \; J = P_\text{loss} \cdot p_\text{loss}^P + \sum_{a \in A} Q_a \cdot p_a^Q(s)`

Maximize Renewable Feed-In
-------------------------------------------
Use :code:`from opfgym.envs import MaxRenewable` to import this environment.
The environment represents a renewable feed-in maximization problem. The goal
is to find the optimal active power setpoints of generators
that maximizes the renewable feed-in while satisfying all constraints.
Additionally, storage systems can be utilized for constraint satisfaction but
are not part of the objective function.

:math:`\text{min} \; J = -\sum_{g \in G} P_g`
