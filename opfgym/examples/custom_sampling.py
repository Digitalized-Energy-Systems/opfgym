
import numpy as np
import pandapower as pp
from scipy.stats import weibull_min

from opfgym import opf_env
import opfgym.sampling as sampling
from opfgym.simbench.build_simbench_net import build_simbench_net
import opfgym.constraints as constraints
from opfgym.simbench.data_split import define_test_train_split


class CustomWindPowerSampler(sampling.StateSampler):
    """ Custom sampler that uses the Weibull distribution for wind power 
    generation"""
    def sample_state(self, net, *args, **kwargs):
        # Weibull distribution
        shape = 2.0
        scale = 10.0
        wind_speeds = weibull_min.rvs(shape, scale=scale, size=1)[0]

        # Assumption: cubic relationship
        relative_power = np.clip((wind_speeds / scale) ** 3, 0, 1)

        # Assumption: All wind turbines are close to each other -> same wind speed for all of them
        idxs = net.sgen["type"].str.contains("wind", case=False, na=False)
        max_power = net.sgen.loc[idxs, 'max_max_p_mw'] / net.sgen.loc[idxs, 'scaling']
        net.sgen.loc[idxs, "p_mw"] = relative_power * max_power

        return net


class CustomSampling(opf_env.OpfEnv):
    def __init__(
            self,
            simbench_network_name='1-LV-urban6--0-sw',
            cos_phi=0.95,
            *args, **kwargs
            ):

        self.cos_phi = cos_phi
        net, profiles = self._define_opf(
            simbench_network_name, *args, **kwargs)

        obs_keys = [
            ('load', 'p_mw', net.load.index),
            ('load', 'q_mvar', net.load.index),
            ('sgen', 'p_mw', net.sgen.index[~net.sgen.controllable])
        ]

        act_keys = [('sgen', 'q_mvar', net.sgen.index[net.sgen.controllable])]

        # Explicitly define the data split into train/validation/test
        simbench_data_split = define_test_train_split(**kwargs)
        # Define the data sampler: Use SimBench data for everything and
        # overwrite with out custom distribution afterwards
        train_sampling = sampling.SequentialSampler(samplers=[
            sampling.SimbenchSampler(
                obs_keys,
                profiles=profiles,
                available_steps=simbench_data_split[0],
                **kwargs
                ),
            # By defining the custom sampler after the SimBench sampler, the
            # SimBench values will be overwritten, which is intentional here.
            CustomWindPowerSampler()
        ])

        super().__init__(net, act_keys, obs_keys, profiles=profiles,
                         train_sampling=train_sampling,
                         simbench_data_split=simbench_data_split,
                         *args, **kwargs)

    def _define_opf(self, simbench_network_name, *args, **kwargs):
        net, profiles = build_simbench_net(
            simbench_network_name, *args, **kwargs)

        # Define first two generators as wind
        net.sgen.type[:2] = 'wind'

        # Control all non-wind-turbine generators
        wind = net.sgen["type"].str.contains("wind", case=False, na=False)
        net.sgen['controllable'] = False
        net.sgen.loc[net.sgen.index[~wind], 'controllable'] = True
        net.sgen['min_p_mw'] = net.sgen['min_min_p_mw']
        net.sgen['max_p_mw'] = net.sgen['max_max_p_mw']
        net.sgen['min_q_mvar'] = 0.0
        net.sgen['max_q_mvar'] = 0.0

        # Set everything else to uncontrollable
        for unit_type in ('load', 'gen', 'storage'):
            net[unit_type]['controllable'] = False

        # Objective: Minimize the active power flow from external grid
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=1)

        return net, profiles



if __name__ == '__main__':
    env = CustomSampling()
    for _ in range(5):
        env.reset()
        env.step(env.action_space.sample())
