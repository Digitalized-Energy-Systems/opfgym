Environment Design Options
==========================

The `OpfEnv` base class provides various environment design options for 
representing the OPF problem as an RL environment. These options are available
to all environments that inherit from the `OpfEnv` base class.

Overall, the RL environment design consists of the following main components:

* Reward function definition
* Observation space definition
* Action space definition
* Episode definition
* Training and test data 

Most environment design options are described in detail in 
`Wolgast and Nieße - Learning the optimal power flow: Environment design matters <https://www.sciencedirect.com/science/article/pii/S2666546824000764>`_.

TODO: Work in progress, more information will follow.

Reward function
---------------

The reward function represents the goal of the agent. In the case of the RL-OPF,
the goal is to minimize the objective function while satisfying all constraints,
which can be represented by penalties. 

Three different standard reward functions to combine the objective function and
the constraint violations are available:

Summation reward
^^^^^^^^^^^^^^^^
In the summation reward, we simply add the penalties :math:`p_i(x)` 
for constraint violations in the current state :math:`x`
to the negative objective function value :math:`f(x)`:

:math:`r = -f(x) - \sum_{i} p_i(x)`

Replacement reward
^^^^^^^^^^^^^^^^^^
In the replacement reward, we only provide either the objective function value
as a learning feedback or the penalty:

If valid: :math:`r = -f(x) + C`

Else: :math:`r = -\sum_{i} p_i(x)`

Additionally, we need a constant :math:`C` to ensure that the valid reward is 
always better than the invalid one.

Parameterized reward
^^^^^^^^^^^^^^^^^^^^
This reward combines the previous two and allows for all possible combinations:

If valid: :math:`r = -f(x) + C_{valid}`

Else: :math:`r = w * -f(x) - \sum_{i} p_i(x) - C_{invalid}`

Note that if the objective weight :math:`w` is set to zero, it is equivalent to
the replacement reward. If it is set to one and both constants 
:math:`C` are set to zero, it is equivalent to the summation reward. 


Observation space
-----------------

TODO: Work in progress, more information will follow.

Action space
------------

TODO: Work in progress, more information will follow.

Episode definition
------------------

TODO: Work in progress, more information will follow.

Training and test data
----------------------

TODO: Work in progress, more information will follow.
