Data Sampling
=============

The StateSampler Class
-----------------------

In the gymnasium :meth:`reset` method, the environment is set to some random 
state. This is done by the :meth:`_sampling` method of the base class, which
calls a :class:`StateSampler` object. In *OPF-Gym*, three standard data samplers
are pre-implemented: The :class:`SimbenchSampler`, the :class:`NormalSampler`,
and the :class:`UniformSampler`. The default in *OPF-Gym* is to use SimBench 
data for active and reactive power values, and a uniform distribution for state
variables that are not included in the SimBench data (e.g. prices, slack 
voltages, etc.).


SimBench Data 
_____________________
.. autoclass:: opfgym.sampling.SimbenchSampler
   :members:

Normal Distribution Data
_______________________________
.. autoclass:: opfgym.sampling.NormalSampler
   :members:

Uniform Distribution Data
_______________________________
.. autoclass:: opfgym.sampling.UniformSampler
   :members:


Data Sampling Wrappers
----------------------

In many cases, we want to combine multiple distributions for different state
variables, for example by sampling generation data from one distribution
and market prices from another. In *OPF-Gym*, this is done with the 
:class:`StateSamplerWrapper` class. Two standard wrappers are pre-implemented:
The :class:`SequentialSampler` and the :class:`MixedRandomSampler`.

The SequentialSampler
_______________________________
.. autoclass:: opfgym.sampling.SequentialSampler
   :members:

The MixedRandomSampler
_______________________________
.. autoclass:: opfgym.sampling.MixedRandomSampler
   :members: