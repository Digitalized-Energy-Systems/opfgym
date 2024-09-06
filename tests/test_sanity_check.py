
# TODO: Somewhat convoluted to write unit tests for a test function. How can this be done better?

import numpy as np
import pandapower as pp

from .sanity_check import check_action_space


def test_check_action_space():

    # Create basic power system
    net = pp.create.create_empty_network()
    pp.create.create_bus(net, vn_kv=20)
    for i in range(3):
        pp.create.create_sgen(net, bus=0, p_mw=1, q_mvar=0, max_p_mw=2, 
                              min_p_mw=1, min_q_mvar=0, max_q_mvar=0, 
                              controllable=False)  


    act_keys = (('sgen', 'p_mw', [0, 1, 2],),)
    try:
        check_action_space(net, act_keys)
        success = True
    except AssertionError as e:
        success = False
    assert not success

    net.sgen['controllable'] = True
    try:
        check_action_space(net, act_keys)
        success = True
    except AssertionError as e:
        success = False
    assert success 

    net.sgen['in_service'] = False
    try:
        check_action_space(net, act_keys)
        success = True
    except AssertionError as e:
        success = False
    assert not success 

    net.sgen['in_service'] = True
    net.sgen['min_q_mvar'] = -1
    net.sgen['max_q_mvar'] = 1
    try:
        check_action_space(net, act_keys)
        success = True
    except AssertionError as e:
        success = False
    assert not success 

    act_keys = (('sgen', 'p_mw', [0, 1, 2],), ('sgen', 'q_mvar', [0, 1, 2],))
    try:
        check_action_space(net, act_keys)
        success = True
    except AssertionError as e:
        success = False
    assert success 

    pp.create.create_load(net, bus=0, p_mw=1, controllable=True)
    try:
        check_action_space(net, act_keys)
        success = True
    except AssertionError as e:
        success = False
    assert not success    

    net.load['controllable'] = False
    try:
        check_action_space(net, act_keys)
        success = True
    except AssertionError as e:
        success = False
    assert success    
