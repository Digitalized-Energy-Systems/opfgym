Benchmarks
==========

This page lists all available RL-OPF benchmarks of opfgym.

Ideally, the benchmarks should be loaded and used for benchmarking of new 
RL-OPF algorithms without changing the underlying OPF problem formulation.
This way, comparability of research results across multiple publications 
is ensured. 

The following table provides an overview of the benchmark environments, 
including the utilized Simbench benchmark power system, the number of buses, 
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

All benchmark problems represent standard OPF problems with different 
objectives and actuators. Advanced OPF problems like multi-stage OPF or the 
security-constrained OPF are not considered (yet). However, they are possible 
in *OPF-Gym* by creating custom environments.

By default all benchmarks consider the voltage band constraints, power balance
constraints, line overload constraints, and transformer loading constraints. 

.. TODO: Maybe add references to the constraints in the documentation.

Economic Dispatch (:code:`EcoDispatch`)
---------------------------------
Use :code:`from opfgym.envs import EcoDispatch` to import this environment.
The environment represents an economic dispatch problem. The goal is to 
find the optimal active power generator setpoints of a given state that 
minimize the generation costs while satisfying the power balance constraints.

Voltage Control (:code:`VoltageControl`)
---------------------------------
Use :code:`from opfgym.envs import VoltageControl` to import this environment.
The environment represents a voltage control problem. The goal is to find 
the optimal reactive power setpoints of generators and a single storage system 
that minimize the overall power losses while satisfying the voltage band
constraints and all other default constraints.

Load Shedding (:code:`LoadShedding`)
------------------------------
Use :code:`from opfgym.envs import LoadShedding` to import this environment.
The environment represents a load shedding problem. The goal is to find the
least amount of load shedding that is required to satisfy all constraints.
All loads are assigned with different priorities, represented by cost codetions.
Further, the agent can utilize storage systems to reduce the amount of load
shedding required.

Reactive Power Market (:code:`QMarket`)
--------------------------------
Use :code:`from opfgym.envs import QMarket` to import this environment.
The environment represents a reactive power market problem, which is an 
extension to the voltage control problem. The goal is again to find optimal 
reactive power setpoints of generators that minimize the overall power losses. 
However, this time, the reactive power providers need to be paid. Therefore,
a trade-off needs to be found between minimizing power losses and minimizing
the reactive power costs.

Maximize Renewable Feed-In (:code:`MaxRenewable`)
-------------------------------------------
Use :code:`from opfgym.envs import MaxRenewable` to import this environment.
The environment represents a renewable feed-in maximization problem. The goal
is to find the optimal active power setpoints of generators and storage systems
that maximizes the renewable feed-in while satisfying all constraints.
