The :class:`OpfEnv` Base Class
==============================

The `OpfEnv` base class is the core of the *OPF-Gym* library. 
It inherits from the :class:`gymnasium.Env` class and implements the gymnasium API. 
Additionally, it provides the following main functionalities:

* Automatic conversion of a pandapower OPF problem to a gymnasium environment.
* Extended API to compute a baseline OPF solution (using pandapower) and
  to evaluate the performance of the RL agent.
* Various pre-implemented environment design options for representing the OPF
  problem as an RL environment. (see :ref:`Environment Design Options`)

All environments that inherit from the :class:`OpfEnv` base class will have these
functionalities. 

Methods:
--------

The `OpfEnv` base class provides the following methods:

.. automethod:: opfgym.OpfEnv.reset

.. automethod:: opfgym.OpfEnv.step

.. automethod:: opfgym.OpfEnv.render 

.. automethod:: opfgym.OpfEnv.get_state 

.. automethod:: opfgym.OpfEnv.run_power_flow

.. automethod:: opfgym.OpfEnv.run_optimal_power_flow

.. automethod:: opfgym.OpfEnv.get_objective

.. automethod:: opfgym.OpfEnv.get_optimal_objective

.. automethod:: opfgym.OpfEnv.is_state_valid

.. automethod:: opfgym.OpfEnv.is_optimal_state_valid

.. automethod:: opfgym.OpfEnv.get_actions

.. automethod:: opfgym.OpfEnv.get_optimal_actions
