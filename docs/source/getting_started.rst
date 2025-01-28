Getting Started
===============

This section provides a brief introduction how to get started with *OPF-Gym*.
The next three sections demonstrate how to load and run a benchmark environment,
how to select environment design options, and how to define a custom RL-OPF
environment.

Loading and running a benchmark environment
-------------------------------------------

The following code loads the reactive power market benchmark environment, runs 
it for three episodes, and computes some performance metrics.

.. code-block:: python

    from opfgym.envs import QMarket

    # Load the reactive power market benchmark environment
    env = QMarket()  

    num_episodes = 3
    for _ in range(num_episodes):
        observation, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated: 
            # Perform random action (replace this with a learning agent!)
            action = env.action_space.sample()  

            # Perform a step according to the gymnasium API
            observation, reward, terminated, truncated, info = env.step(action)

            # Check for constraint satisfaction
            print(f"The grid satisfies all constraints: {env.is_state_valid()}")

            # Perform conventional OPF for comparison
            success = env.run_optimal_power_flow()
            if not success:
                print("The OPF calculation failed. Metrics cannot be computed.")
                continue

            # Compute the error compared to the optimal solution
            objective = sum(env.calculate_objective())
            optimal_objective = env.get_optimal_objective()
            optimal_actions = env.get_optimal_actions()
            percentage_error = optimal_objective - objective / abs(optimal_objective) * 100
            print(f"Optimization error of the random action: {round(percentage_error, 2)}%")
            print(f"Optimal actions: {optimal_actions[:3]} (first three entries)")
            print(f"Agent actions: {action[:3]} (first three entries)")
            print("-------------------------------------")


Adjusting the environment design options
----------------------------------------

The following code demonstrates how to utlize the pre-implemented environment 
design options:

.. code-block:: python

    from opfgym.envs import QMarket

    # Define some exemplary environment design options
    kwargs = {
        # Add the voltage magnitudes and angles to the observation space
        'add_res_obs': ('voltage_magnitude', 'voltage_angle'),
        # Sample the training data uniformly in the full possible data range.
        'train_data': 'full_uniform',
        # Sample the test data from the SimBench time-series data.
        'test_data': 'simbench',
        # Prioritize constraint satisfaction over optimization performance.
        'penalty_weight': 0.8,
    }

    # Load the reactive power market benchmark environment
    env = QMarket(**kwargs)  

    # Interact with the environment in the usual way (see above)
    obs, info = env.reset()
    # ...

For more information on environment design and why it is important, see
`Wolgast and Nieße - Learning the optimal power flow: Environment design matters <https://www.sciencedirect.com/science/article/pii/S2666546824000764>`_.

.. The full list of pre-implemented environment design options can be found in 
.. :ref:`Environment Design Options`.

Note that the environment design options do not change the underlying OPF
problem formulation. They only change the representation of the OPF as an
RL environment. This way, they simplify or complicate the learning problem for
the agent, but do not change the OPF problem itself.


Define a custom RL-OPF environment
----------------------------------

The following code demonstrates how to define a custom RL-OPF environment by 
inheritance from the base class :class:`OpfEnv`. The key aspects are to define the
pandapower network and its OPF problem, and the observation and action spaces 
by simply selecting which pandapower table entries to use as observation/action.
More details can be found in :ref:`Create Custom Environments`.

.. code-block:: python

    from opfgym import OpfEnv
    from opfgym.simbench.build_simbench_net import build_simbench_net

    class CustomEnv(OpfEnv):
        def __init__(self, **kwargs):

            net, profiles = self._define_opf()

            # Define the observation space by providing the keys to the 
            # respective pandapower tables and columns to observe
            # (automatically transformed into a gymnasium space)
            obs_keys = (
                # Observe all loads active and reactive power
                ('load', 'p_mw', net.load.index),
                ('load', 'q_mvar', net.load.index),
                # The structure is always (unit_type, column_name, unit_indexes)
            )

            # Define the action space in the same way
            act_keys = (
                # Control all sgens' active power
                ('sgen', 'p_mw', net.sgen.index),
            )

            super().__init__(net, act_keys, obs_keys, profiles=profiles, **kwargs)

        def _define_opf(self):
            """ Define the OPF problem in a pandapower net. """

            # Load a simbench network, including time-series data profiles
            net, profiles = build_simbench_net('1-LV-urban6--0-sw')

            # Set sgens as controllable
            net.sgen['controllable'] = True
            net.sgen['min_p_mw'] = 0
            net.sgen['max_p_mw'] = 1
            # Set reactive power as uncontrollable by restricting it to zero
            net.sgen['min_q_mvar'] = 0
            net.sgen['max_q_mvar'] = 0

            # Set everything else to uncontrollable explicitly
            for unit_type in ('load', 'gen', 'storage'):
                net[unit_type]['controllable'] = False

            # Define minimal objective function by setting costs
            for idx in net.ext_grid.index:
                pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=1)

            return net, profiles

    # Note that by inheriting from `OpfEnv`, all standard env design options are available
    kwargs = {
        # Add current line load to the observation space
        'add_res_obs': ['line_loading'],
        # ...
    }

    # Load the custom environment
    env = CustomEnv(**kwargs)

    # Interact with the environment in the usual way (see above)
    obs, info = env.reset()
    # ...
