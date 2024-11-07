import pytest

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

import opfgym.data_sampling as sampling


@pytest.fixture
def net():
    return pn.example_simple()

def test_base_class(net):
    class MockSampler(sampling.DatasetSampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def sample_state(self, net):
            return net

    # Test seeding
    sampler1 = MockSampler(seed=42)
    sampler2 = MockSampler(seed=42)
    assert sampler1.np_random.rand() == sampler2.np_random.rand()
    assert sampler1.np_random.rand() == sampler2.np_random.rand()
    sampler2.set_seed(43)
    assert sampler1.np_random.rand() != sampler2.np_random.rand()

    # Test hooks
    def hook1(net):
        net.sgen.p_mw[0] = 1.0
        return net

    def hook2(net):
        net.sgen.p_mw[0] = 2.0
        return net

    def hook3(net):
        net.test_variable = 'success'
        return net


    sampler = MockSampler(after_sampling_hooks=[hook1])
    net = sampler(net)
    assert net.sgen.p_mw[0] == 1.0

    # Hooks are called in the order they are given -> hook2 overwrites hook1
    sampler = MockSampler(after_sampling_hooks=[hook1, hook2])
    net = sampler(net)
    assert net.sgen.p_mw[0] == 2.0

    # Test if hooks can perform arbitrary operations on the net
    sampler = MockSampler(after_sampling_hooks=[hook3])
    net = sampler(net)
    assert net.test_variable == 'success'

def test_simbench_sampling(net):
    df = pd.DataFrame(np.arange(3), index=np.arange(3))
    profiles = {('sgen', 'p_mw'): df}
    sample_keys = [('sgen', 'p_mw', np.array([0]))]

    sampler = sampling.SimbenchSampler(sample_keys=sample_keys,
                                       profiles=profiles)
    net = sampler(net)
    assert net.sgen.p_mw[0] in [0, 1, 2]

    net = sampler(net, step=1)
    assert net.sgen.p_mw[0] == 1

    net = sampler(net, step=2)
    assert net.sgen.p_mw[0] == 2

def test_normal_sampling(net):
    sample_keys = [('sgen', 'p_mw', np.array([0]))]
    sampler = sampling.NormalSampler(sample_keys=sample_keys)

    # Mean and std not yet defined -> should raise an error
    with pytest.raises(KeyError):
        net = sampler(net)

    # Define mean and std -> should not raise an error anymore
    net.sgen['mean_p_mw'] = 0
    net.sgen['std_dev_p_mw'] = 2
    net.sgen['min_p_mw'] = -1
    net.sgen['max_p_mw'] = 1
    net.sgen['p_mw'] = np.nan
    net = sampler(net)
    assert ~np.isnan(net.sgen.p_mw[0])
    assert -1 <= net.sgen.p_mw[0] <= 1

    # Should always be clipped to allowed data range
    net.sgen['min_p_mw'] = 0
    net.sgen['max_p_mw'] = 0
    net = sampler(net)
    assert net.sgen.p_mw[0] == 0

def test_uniform_sampling(net):
    sample_keys = [('sgen', 'p_mw', np.array([0]))]
    sampler = sampling.UniformSampler(sample_keys=sample_keys)

    # Min/max range not yet defines -> should raise an error
    with pytest.raises(KeyError):
        net = sampler(net)

    # Define min/max range -> should not raise an error anymore
    net.sgen['min_p_mw'] = 0
    net.sgen['max_p_mw'] = 0.1
    net = sampler(net)
    assert 0 <= net.sgen.p_mw[0] <= 0.1

    # If full data range is defined, that range should be used instead of the pandapower OPF constraints
    net.sgen['min_min_p_mw'] = 0.1
    net.sgen['max_max_p_mw'] = 0.2
    net = sampler(net)
    assert 0.1 <= net.sgen.p_mw[0] <= 0.2

def test_sequential_sampling(net):
    sample_keys1 = [('sgen', 'p_mw', np.array([0]))]
    sample_keys2 = [('load', 'p_mw', np.array([0]))]
    samplers = sampling.UniformSampler(sample_keys=sample_keys1), sampling.NormalSampler(sample_keys=sample_keys2)
    sampler = sampling.SequentialSampler(samplers)

    assert len(sampler) == 2
    assert isinstance(sampler[0], sampling.UniformSampler)

    # Min/max range not yet defines -> should raise an error
    with pytest.raises(KeyError):
        net = sampler(net)

    # Define min/max range -> should not raise an error anymore
    net.sgen['min_p_mw'] = 0
    net.sgen['max_p_mw'] = 0.1
    net.sgen['mean_p_mw'] = 0
    net.sgen['std_dev_p_mw'] = 2

    net.load['scaling'] = 1
    net.load['min_p_mw'] = -0.1
    net.load['max_p_mw'] = 0
    net.load['mean_p_mw'] = 0
    net.load['std_dev_p_mw'] = 2

    net = sampler(net)
    assert 0 <= net.sgen.p_mw[0] <= 0.1
    assert -0.1 <= net.load.p_mw[0] <= 0

def test_mixed_sampling(net):
    sample_keys = [('sgen', 'p_mw', np.array([0]))]
    profiles = {('sgen', 'p_mw'): pd.DataFrame(np.arange(2), index=np.arange(2))}
    sampler_probabilities_cumulated = [0.5, 0.75, 1.0]

    # Len of probabilities should be equal to number of samplers
    with pytest.raises(AssertionError):
        sampler = sampling.StandardMixedRandomSampler(
            sample_keys=sample_keys,
            profiles=profiles,
            sampler_probabilities_cumulated=sampler_probabilities_cumulated[:2])

    # Profiles are strictly required
    with pytest.raises(TypeError):
        sampler = sampling.StandardMixedRandomSampler(
            sample_keys=sample_keys,
            # profiles=profiles,
            sampler_probabilities_cumulated=sampler_probabilities_cumulated)

    sampler = sampling.StandardMixedRandomSampler(
        sample_keys=sample_keys,
        profiles=profiles,
        sampler_probabilities_cumulated=sampler_probabilities_cumulated)

    net.sgen['min_p_mw'] = 0
    net.sgen['max_p_mw'] = 1
    net.sgen['mean_p_mw'] = 0
    net.sgen['std_dev_p_mw'] = 2

    assert 0 <= sampler(net).sgen.p_mw[0] <= 1
    assert 0 <= sampler(net).sgen.p_mw[0] <= 1
    assert 0 <= sampler(net).sgen.p_mw[0] <= 1