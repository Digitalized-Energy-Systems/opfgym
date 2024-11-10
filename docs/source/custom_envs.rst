Create Custom Environments
===========================

By inheriting from the :code:`OpfEnv` base class, a wide variety of custom 
environments can be created. In the process, some steps have to be considered.

For a full example, refer to :ref:`Define a custom RL-OPF environment`.

Class Initialization
--------------------
In the :code:`__init__` method, multiple attributes have to be set to define 
the OPF problem before calling :code:`super().__init__()`. 
The relevant attributes are:

* :code:`self.net` - The pandapower network object, including constraints.
* :code:`self.obs_keys` - The observation space definition.
* :code:`self.act_keys` - The action space definition.

.. code-block:: python

    from opfgym import OpfEnv

    class CustomEnv(OpfEnv):
        def __init__(self):
            # Define the pandapower network
            self.net = ...
            # Define the observation space
            self.obs_keys = ...
            # Define the action space
            self.act_keys = ...

            super().__init__()

Pandapower network definition
-----------------------------
The pandapower network object can be created from scratch or pre-implemented 
networks can be  loaded from 
`pandapower <https://pandapower.readthedocs.io/en/latest/networks.html>`_ 
or `simbench <https://simbench.readthedocs.io/en/stable/networks.html>`_. 

Load from pandapower:

.. code-block:: python

    import pandapower.networks as pn
    net = pn.case_ieee30() # Load IEEE 30 bus network

Create a network from scratch:

.. code-block:: python

    import pandapower as pp
    net = pp.create_empty_network()
    # Add buses, lines, loads, etc. to the network, see pandapower docs

Load from SimBench:

.. code-block:: python

    import simbench as sb
    net = sb.get_simbench_net("1-LV-rural1--0-sw")
    # Optional for data sampling, define the profiles attribute (only for SimBench)
    profiles = sb.get_absolute_values(
        net, profiles_instead_of_study_cases=True)

After creating the network, define the constraints of the OPF problem, 
in standard pandapower practice. 
(compare `pandapower docs <https://pandapower.readthedocs.io/en/latest/opf/formulation.html>`_).

.. code-block:: python 

    net.bus['min_vm_pu'] = 0.95
    net.bus['max_vm_pu'] = 1.05
    net.line['max_loading_percent'] = 100
    net.trafo['max_loading_percent'] = 100

All standard pandapower constraints are automatically considered in the reward
calculation, if defined.

Observation space definition
-----------------------------

The observation space is defined by selecting which pandapower tables, columns, 
and elements the agent can observe. That is done by setting the 
`env.obs_keys` attribute of the environment.

The `obs_keys` are a list of tuples, where each tuple contains the three elements 
`(unit_type, column, indices)`. The `unit_type` is the pandapower table name,
the `column` is the column name of the table, and `indices` is an array of indices
of the elements in the table that the agent can manipulate.

.. code-block:: python

    # Some example observations
    obs_keys = [
        # Observe all load's active and reactive power values
        ('load', 'p_mw', net.load.index),
        ('load', 'q_mvar', net.load.index),
    ]

Additionally, it is required to define the observation ranges. This is required
to create a bounded observation space but also for sampling. To distinguish 
these range definitions from the pandapower OPF constraints, use the 
prefixes :code:`'min_min_'` and :code:`'max_max_'` in front of the column name.

.. code-block:: python

    # Example observation ranges
    net.load['min_min_p_mw'] = 0
    net.load['max_max_p_mw'] = 100
    net.load['min_min_q_mvar'] = -50
    net.load['max_max_q_mvar'] = 50

Action space definition
-----------------------

The action space is defined by selecting which pandapower tables, columns, 
and elements the agent can manipulate. That is done by setting the 
`env.act_keys` attribute of the environment.

Exactly as for the observation keys,  `act_keys` are a list of tuples, 
where each tuple contains the three elements 
`(unit_type, column, indices)`.

.. code-block:: python

    import numpy as np

    # Some example actions
    act_keys = [
        # Control active power of all generators
        ('sgen', 'p_mw', net.sgen.index)
        # Control tap changer of transformer 0
        ('trafo', 'tap_pos', np.array([0]))
        # Control the status of line 1
        ('switch', 'closed', np.array([1]))
    ]

Additionally, it is required that for each defined actuator, the corresponding 
action ranges are defined, which happens exactly as in pandapower. Simply add 
the :code:`'min_'`/:code:`'max_'` prefix to the column name and set the values 
as desired.

.. code-block:: python

    # Example action ranges
    net.sgen['min_p_mw'] = 0
    net.sgen['max_p_mw'] = 100
    # Discrete and boolean actions are possible, too, and get recognized automatically
    net.trafo['min_tap_pos'] = -2
    net.trafo['max_tap_pos'] = 2
    net.switch['min_closed'] = 0
    net.switch['max_closed'] = 1

Further, make sure to set the units as controllable in the pandapower net.
It is good practive to set all other units as not controllable explicitly to
prevent errors.

.. code-block:: python

    net.sgen['controllable'] = True
    # Only trafo 0 is controllable
    net.trafo['controllable'] = False
    net.trafo['controllable'][0] = True
    # Only switch 1 is controllable
    net.switch['controllable'] = False
    net.switch['controllable'][1] = True


Advanced concepts   
-----------------

In some cases, it is required to implement advanced features, which require to 
overwrite some methods of the base class.

To set dynamic constraints that change with the state of the power network, 
overwrite the :meth:`_sampling` method.

.. code-block:: python

    def _sampling(self):
        # Define dynamic constraints, for example, fixing reactive power of 
        # sgens to the current state so that pandapower OPF does not use them 
        # as control variables
        self.net.sgen['min_q_mvar'] = self.net.sgen.q_mvar 
        self.net.sgen['max_q_mvar'] = self.net.sgen.q_mvar

If your OPF problem is not solvable with the standard pandapower OPF solver, 
provide a :func:`optimal_power_flow_solver` function to the base class 
:meth:`__init__` method. The same is possible for the power flow solver.

.. code-block:: python

    class CustomEnv(OpfEnv):
        def __init__(self):
            ...
            def custom_opf_solver(net, **kwargs):
                # Custom power flow solver
                ...

            def custom_power_flow_solver(net, **kwargs):
                # Custom power flow solver
                ...

            super().__init__(optimal_power_flow_solver=custom_opf_solver,
                             power_flow_solver=custom_power_flow_solver)
