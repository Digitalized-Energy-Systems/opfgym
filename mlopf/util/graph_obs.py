
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def get_bus_aggregated_obs(net, unit_type, column, idxs, zero_data=False):
    """ Aggregate power values that are connected to the same bus to reduce
    state space. """
    df = net[unit_type].iloc[idxs]
    print(unit_type)
    bus_data = net[unit_type[4:] if 'res' in unit_type else unit_type].bus
    data = df.groupby(by=bus_data)[column].sum()
    if zero_data:
        # Keep zero entries
        all_data = pd.Series(np.zeros(len(net.bus)), index=net.bus.index)
        all_data[data.index] = data
        return all_data.to_numpy()
    return data.to_numpy()


def get_homo_graph_obs(net, obs_keys):
    """ Create homogenous observation data for pytorch geometric and GNNs """

    n_nodes = len(net.bus)

    n_node_attr = 0
    for unit_type, column, idxs in obs_keys:
        if unit_type in ('res_bus', 'load', 'sgen', 'poly_cost', 'res_ext_grid'):
            n_node_attr += 1

    node_attr = np.zeros((n_nodes, n_node_attr))
    node_counter = 0
    edge_attr = None
    for unit_type, column, idxs in obs_keys:
        # Add node attributes for the graph
        if unit_type in ('res_bus', 'load', 'sgen', 'poly_cost', 'res_ext_grid'):
            if unit_type == 'res_bus':
                node_attr[:, node_counter] = net.res_bus[column].to_numpy()
            elif unit_type in ('load', 'sgen', 'res_ext_grid', 'poly_cost'):
                # Warning: For cost data, a homogenous graph does not make sense!
                # However: Will work as long as there is only one unit per bus
                node_attr[:, node_counter] = get_bus_aggregated_obs(
                    net, unit_type, column, idxs, zero_data=True) 
            node_counter += 1
        # Add edge attributes for the graph
        if unit_type in ('res_line', 'res_trafo', 'res_trafo3w'):
            edge_attr = np.concatenate((net['res_line'][column].to_numpy(), 
                                        net['res_trafo'][column].to_numpy(), 
                                        net['res_trafo3w'][column].to_numpy()), 
                                        axis=0)   
            edge_attr = np.reshape(edge_attr, (len(edge_attr), 1))

    # Get graph connectivity data (TODO: Only required once if topology constant!)
    edge_index = get_edge_index(net)

    # Convert to pytorch tensor data
    node_attr = torch.tensor(node_attr, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    if edge_attr is not None:
        edge_attr = np.concatenate((edge_attr, edge_attr), axis=0)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    print(Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr))

    return Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)


def get_edge_index(net):
    n_edges = len(net.line) + len(net.trafo) + len(net.trafo3w)

    line_connections = net.line[['from_bus', 'to_bus']].to_numpy()
    trafo_connections = net.trafo[['hv_bus', 'lv_bus']].to_numpy()
    edge_index = np.concatenate((line_connections, trafo_connections), axis=0)
    edge_index = np.reshape(edge_index, (2, n_edges))

    # Convert to undirected graph data 
    edge_index = np.concatenate((edge_index, edge_index[::-1,:]), axis=1)

    return edge_index


def get_simple_graph_obs(net):
    """ Create simple observation data for pytorch geometric and GNNs.
    Neglects prices, voltage angles and line attributes. """

    # Prepare network (TODO Move to extra function)
    # Extreme case: Max bus power = sum of all max unit powers at that bus 
    # # TODO: Current assumptions: All positive, no storages, etc.
    # net.bus['max_max_p_mw'] = 0
    # for unit_type in ('load', 'sgen', 'ext_grid'):
    #     if hasattr(net[unit_type], 'max_max_p_mw'):
    #         max_p_mw = net.sgen.groupby(by=net[unit_type].bus)['max_max_p_mw'].sum()
    #         net.bus['max_max_p_mw'][max_p_mw.index] += max_p_mw.values
        
    # net.bus['max_max_q_mvar'] = 0
    # for unit_type in ('load', 'sgen', 'ext_grid'):
    #     if hasattr(net[unit_type], 'max_max_q_mvar'):
    #         max_q_mvar = net.sgen.groupby(by=net[unit_type].bus)['max_max_q_mvar'].sum()
    #         net.bus['max_max_q_mvar'] += max_p_mw
    # import pdb; pdb.set_trace()
    n_node_attr = 3  # Active and reactive power + voltage magnitude (TODO angle?)
    node_attr = np.zeros((len(net.bus), n_node_attr))
    # scaler = net.bus.max_max_p_mw.copy()
    # scaler[scaler==0.0] = 1.0
    node_attr[:, 0] = net.res_bus.p_mw.values # / scaler
    node_attr[:, 1] = net.res_bus.q_mvar.values
    # Voltage magnituded can be scaled directly!
    node_attr[:, 2] = (net.res_bus.vm_pu.values - 1.0) * 10

    edge_index = get_edge_index(net)

    # Convert to pytorch tensor data
    node_attr = torch.tensor(node_attr, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return Data(x=node_attr, edge_index=edge_index)
