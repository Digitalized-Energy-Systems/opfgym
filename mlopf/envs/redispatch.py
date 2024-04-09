""" The redispatch problem as RL-OPF environment. """

import numpy as np
import pandapower as pp

from mlopf import opf_env
from mlopf.build_simbench_net import build_simbench_net


class Redispatch(opf_env.OpfEnv):
    """
    Redispatch: The grid has an existing constraint violation that needs
    to be resolved. The grid operator attempts to resolve the violation without
    changing the overall sum of power in the system (slack active power fixed). 
    The grid operator has to increase feed-in of some generators and decrease 
    feed-in of others.  

    Actuators: Active power (TODO diff?) of all gens and sgens; 
        TODO: Maybe some controllable loads/storages as well?

    Sensors: 
        active+reactive power of all loads; 
        active power of all generation units;
        active power diff prices of all generation units; 

    Objective: minimize redispatch costs (equals: min weighted active power changes in the system)

    Constraints: Voltage band, line/trafo load
    """

    def __init__(self, simbench_network_name='1-EHV-mixed--0-sw',
                 gen_scaling=1.0, load_scaling=0.7, seed=None,
                 cos_phi=0.9, *args, **kwargs):

        self.cos_phi = cos_phi
        self.net = self._define_opf(
            simbench_network_name, 
            gen_scaling=gen_scaling,
            load_scaling=load_scaling, 
            *args, **kwargs)

        # Define the RL problem
        # See all load power values, sgen active power, and gen prices...
        self.obs_keys = [
            ('res_gen', 'p_mw', self.net.gen.index),
            ('res_sgen', 'p_mw', self.net.sgen.index),
            ('load', 'p_mw', self.net.load.index),
            ('load', 'q_mvar', self.net.load.index),
            ('gen', 'diff_price_eur_per_mw', self.net.gen.index),
            ('sgen', 'diff_price_eur_per_mw', self.net.sgen.index),
            ('ext_grid', 'diff_price_eur_per_mw', self.net.ext_grid.index),
        ]

        # ... and control all generator active power values
        self.act_keys = [('gen', 'p_mw', self.net.gen.index),
                         ('sgen', 'p_mw', self.net.sgen.index)]

        if 'ext_grid_pen_kwargs' not in kwargs:
            kwargs['ext_grid_pen_kwargs'] = {'linear_penalty': 500}
        
        super().__init__(seed=seed, pf_for_obs=False, *args, **kwargs)

        # TODO
        if self.vector_reward is True:
            # 2 penalties and `n_sgen+1` objective functions
            n_objs = 2 + len(self.net.sgen) + 1
            self.reward_space = gym.spaces.Box(
                low=-np.ones(n_objs) * np.inf, high=np.ones(n_objs) * np.inf, seed=seed)

    def _define_opf(self, simbench_network_name, voltage_band=0.05, *args, **kwargs):
        net, self.profiles = build_simbench_net(
            simbench_network_name, voltage_band=voltage_band, *args, **kwargs)

        net.gen['vm_pu'] = 1.0 
        net.ext_grid['vm_pu'] = 1.0
        net.ext_grid.drop(np.array([0,1,2,3,4,5]), inplace=True)

        net.load['controllable'] = False

        net.gen['controllable'] = True
        net.gen['max_p_mw'] = self.profiles[('gen', 'p_mw')].max()
        net.gen['min_p_mw'] = self.profiles[('gen', 'p_mw')].min()

        net.sgen['controllable'] = True
        net.sgen['min_p_mw'] = 0
        net.sgen['min_q_mvar'] = 0
        net.sgen['max_q_mvar'] = 0
        net.sgen['q_mvar'] = 0

        # Add price params to the network (as pwl costs so that the OPF works)
        # Init with None to overwrite dynamically at each step
        for idx in net.sgen.index:
            pp.create_pwl_cost(net, idx, 'sgen', points=None)

        for idx in net.gen.index:
            pp.create_pwl_cost(net, idx, 'gen', points=None)

        for idx in net.ext_grid.index:
            pp.create_pwl_cost(net, idx, 'ext_grid', points=None)
        
        # TODO: Instead define pwl costs every step

        # Define range from which to sample active power prices
        self.max_price = 600
        # TODO: Maybe differentiate between increasing and decreasing prices?!
        # Assumption: All price ranges are equal 
        net.sgen['min_diff_price_eur_per_mw'] = 0
        net.sgen['diff_price_eur_per_mw'] = 0
        net.sgen['max_diff_price_eur_per_mw'] = self.max_price
        net.gen['min_diff_price_eur_per_mw'] = 0
        net.gen['diff_price_eur_per_mw'] = 0
        net.gen['max_diff_price_eur_per_mw'] = self.max_price
        net.ext_grid['min_diff_price_eur_per_mw'] = 0
        net.ext_grid['diff_price_eur_per_mw'] = 0
        net.ext_grid['max_diff_price_eur_per_mw'] = self.max_price

        return net

    def _sampling(self, step, test, sample_new, *args, **kwargs):
        super()._sampling(step, test, sample_new, *args, **kwargs)

        # Set sgen max power to current power because RES (TODO: Exception biomass?!)
        # That also means sgen active power can only be reduced not increased
        self.net.sgen.max_p_mw = self.net.sgen.p_mw * self.net.sgen.scaling

        # Sample redispatch prices for all generation units
        self._sample_from_range(
            'sgen', 'diff_price_eur_per_mw', self.net.sgen.index)
        self._sample_from_range(
            'gen', 'diff_price_eur_per_mw', self.net.gen.index)
        self._sample_from_range(
            'ext_grid', 'diff_price_eur_per_mw', self.net.ext_grid.index)

        # Run power flow calculation to get initial active power setpoints
        success = self._run_pf()
        if not success:
            import pdb; pdb.set_trace()

        # Set piece wise linear costs for all generation units
        # Structure: [[min_power, current_power, -costs], [current_power, max_power, costs]]
        min_power = np.concatenate((
            self.net.sgen.min_p_mw.values,
            self.net.gen.min_p_mw.values,
            self.net.ext_grid.min_p_mw.values))
        current_power = np.concatenate((
            self.net.res_sgen.p_mw.values,
            self.net.res_gen.p_mw.values,
            self.net.res_ext_grid.p_mw.values))
        max_power = np.concatenate((
            self.net.sgen.max_p_mw.values,
            self.net.gen.max_p_mw.values,
            self.net.ext_grid.max_p_mw.values))
        price = np.concatenate((
            self.net.sgen.diff_price_eur_per_mw.values,
            self.net.gen.diff_price_eur_per_mw.values,
            self.net.ext_grid.diff_price_eur_per_mw.values))
        downwards_price = list(zip(min_power, current_power, -price))
        upwards_price = list(zip(current_power, max_power, price))
        pwl_price = list(zip(downwards_price, upwards_price))
        # Convert to list of lists as required by pandapower (is this even a problem?)
        pwl_price = list(map(lambda x: list(map(list, x)), pwl_price))
        self.net.pwl_cost['points'] = pwl_price
        # Define constant offset costs so that costs for current power are 0
        self.offset_costs = (current_power * price).mean() 

    def calc_objective(self, net):
        """ Define what to do in vector_reward-case. """
        objs = super().calc_objective(net)
        # Remove offset costs that agent cannot influence anyway
        objs -= self.offset_costs
        if self.vector_reward:
            # Structure: [sgen1_costs, sgen2_costs, ..., loss_costs]
            # TODO: 
            pass
        else:
            return objs

    def calc_violations(self):
        """ Define what to do in vector_reward-case. """
        # Attention: This probably works only for the default system '1-LV-urban6--0-sw'
        # because only ext_grid q violations there and nothing else
        valids, violations, perc_violations, penalties = super().calc_violations()
        if self.vector_reward:
            pass

        # # Additional constraint: Total feed-in must be bigger or equal to before
        # TODO: Not relevant because slack will balance this anyway?! -> Add redispatch costs to slack
        # total_p_mw = net.res_gen.p_mw.sum() + net.res_sgen.p_mw.sum() + net.res_ext_grid.p_mw.sum() - net.res_storage.p_mw.sum()
        # valids.append(total_p_mw >= self.min_total_p_mw) 

        return valids, violations, perc_violations, penalties
    
    # def _run_pf(self, distributed_slack=True, **kwargs):
    #     # Multiple Slack buses in this net! -> overwrite default setting
    #     return super()._run_pf(distributed_slack=distributed_slack, **kwargs)
