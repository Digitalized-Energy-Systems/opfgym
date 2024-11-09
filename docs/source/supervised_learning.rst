Support for Supervised Learning
==========

While the focus of *OPF-Gym* is on reinforcement learning and its environments, 
it also enables comparability with other machine learning approaches like 
supervised or unsupervised learning. 

For that, we provide a convenience function to create a labeled dataset for
supervised learning from any given *OPF-Gym* environment.

.. code-block:: python

    from opfgym.util import create_labeled_dataset
    from opfgym.envs import QMarket  # Or some other environment

    env = QMarket()
    # Set `store_to_path = path` to store the dataset to a directory instead
    inputs, outputs, optimal_objs = create_labeled_dataset(env, num_samples=10)

    # Train your supervised learning algorithm with inputs and outputs
    # supervised_model = train(inputs, outputs)

    # Assuming no data re-scaling, the outputs can directly be fed back to `env.step()`.
    # obs, info = env.reset()
    # action = supervised_model.predict(obs)
    # env.step(action)

Disclaimer: The dataset creation works only for environments that are solvable 
with the pandapower conventional OPF to generate ground-truth labels. That is 
the case for all provided :ref:`Benchmarks`. However, it might not be the case
for custom environments, especially when implementing advanced OPF concepts 
like multi-stage OPF or stochastic OPF. These are not solvable with the 
pandapower OPF. In that case, you have to overwrite the 
:py:meth:`env.run_optimal_power_flow` method of your custom environment and 
provide your own OPF solver.
