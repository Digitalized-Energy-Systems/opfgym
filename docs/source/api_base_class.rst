The :class:`OpfEnv` Base Class
=======================

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

.. TODO: Add list of methods

.. Methods
.. -------

.. The `OpfEnv` base class provides the following methods:


