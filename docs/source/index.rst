.. OPF-Gym documentation master file, created by
   sphinx-quickstart on Fri Nov  8 09:16:51 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


OPF-Gym
=======

*OPF-Gym* is a Python library that provides reinforcement learning (RL) environments 
for learning the optimal power flow (OPF) problem. 

It has three main contributions:

* Five benchmark environments for the RL-OPF problem to enable comparability of research. (see :ref:`Benchmarks`)
* Various pre-implemented environment design options for representing the OPF problem as an RL environment. (see :ref:`Environment Design Options`)
* Convenient creation of custom RL-OPF environments with advanced features like multi-stage OPF, security-constrained OPF, etc. (see :ref:`Create Custom Environments`)

The library uses the `gymnasium <https://gymnasium.farama.org/>`_ RL environment API, 
`pandapower <https://pandapower.readthedocs.io/en/latest/>`_ 
for grid modelling and power flow calculations, 
and integrates the `SimBench <https://simbench.readthedocs.io/en/stable/>`_ 
benchmarks power grids and time-series data by default. 

All pandapower OPF variants can be represented as an RL environment by 
*OPF-Gym*. Additionally, advanced OPF problems like multi-stage OPF,
security-constrained OPF, mixed continuous and discrete actions, stochastic OPF,
etc. are possible as well.

Contact thomas.wolgast@uol.de for questions, feedback, and collaboration.

The repository can be found on 
`GitHub <https://github.com/Digitalized-Energy-Systems/opfgym>`_.

--------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   getting_started
   benchmarks
   api_base_class
   environment_design
   sampling 
   custom_envs
   advanced_features
   supervised_learning

