from collections import deque

import numpy as np
from graph_nets import graphs
from graph_nets import utils_tf

from ..data_utils import indices_to_hot, is_nonetype
from ..dc_opf import GridDCOPF


def print_graphs_tuple(graphs_tuple, verbose=False):
    if verbose:
        print("Shapes of `GraphsTuple`'s fields:")

    print(
        utils_tf.get_num_graphs(graphs_tuple),
        graphs_tuple.map(
            lambda x: x if x is None else x.shape, fields=graphs.ALL_FIELDS
        ),
    )

    if verbose:
        print("\nData contained in `GraphsTuple`'s fields:")
        print("globals:\n{}".format(graphs_tuple.globals))
        print("nodes:\n{}".format(graphs_tuple.nodes))
        print("edges:\n{}".format(graphs_tuple.edges))
        print("senders:\n{}".format(graphs_tuple.senders))
        print("receivers:\n{}".format(graphs_tuple.receivers))
        print("n_node:\n{}".format(graphs_tuple.n_node))
        print("n_edge:\n{}".format(graphs_tuple.n_edge))


def obs_to_graph_dict_by_grid_buses(obs, grid):
    """
    Convert observation to a graph network data dictionary.

    Data dictionary consists of:
        globals: np.ndarray (n_globals, )
            Represents global attributes of a graph.

        nodes: np.ndarray (n_nodes, n_node_features)
            Represents feature vector for each node in the graph.

        edges: np.ndarray (n_edges, n_edge_features)
            Represents feature vector for each edge in the graph.


    Observation attributes are:
        year, month, day, hour_of_day, minute_of_hour, day_of_week:
            Included in globals.

        prod_p, prod_q, prod_v:
            Included in node features.

        load_p, load_q, load_v:
            Included in node features.

        p_or, q_or, v_or, a_or:
            Included in edge features. v_or excluded.

        p_ex, q_ex, v_ex, a_ex:
            Included in edge features. v_ex excluded.

        rho:
            Included in edge features.

        topo_vect:
            Included in edge connections.

        line_status:
            Included as a mask on edge values.

        time_before_cooldown_sub:
            Not included. Can be added to node features.

        time_before_cooldown_line, timestep_overflow, time_next_maintenance, duration_next_maintenance:
            Not included. Can be added to edge features.

        target_dispatch, actual_dispatch:
            Not included.

    """
    n_sub = grid.n_sub
    n_line = grid.n_line
    n_gen = grid.n_gen
    n_load = grid.n_load

    global_features = np.array([obs.month, obs.day, obs.day_of_week, obs.hour_of_day])

    senders_or = grid.line.bus_or.values
    receivers_or = grid.line.bus_ex.values
    edges_or = np.zeros((n_line, 4))
    edges_or[:, 0] = obs.p_or
    edges_or[:, 1] = obs.q_or
    edges_or[:, 2] = obs.a_or
    edges_or[:, 3] = obs.rho
    edges_or = np.multiply(edges_or, np.atleast_2d(obs.line_status.astype(np.float)).T)

    senders_ex = grid.line.bus_ex.values
    receivers_ex = grid.line.bus_or.values
    edges_ex = np.zeros((n_line, 4))
    edges_ex[:, 0] = obs.p_ex
    edges_ex[:, 1] = obs.q_ex
    edges_ex[:, 2] = obs.a_ex
    edges_ex[:, 3] = obs.rho
    edges_ex = np.multiply(edges_ex, np.atleast_2d(obs.line_status.astype(np.float)).T)

    edge_features = np.concatenate((edges_or, edges_ex), axis=0)
    senders = np.concatenate((senders_or, senders_ex), axis=0)
    receivers = np.concatenate((receivers_or, receivers_ex), axis=0)

    node_features = np.zeros((2 * n_sub, (1 + 2 * (n_gen + n_load))))
    for bus_id in grid.bus.index:
        lines_v = [obs.v_or[line_id] for line_id in grid.bus.line_or[bus_id]] + [
            obs.v_ex[line_id] for line_id in grid.bus.line_ex[bus_id]
        ]

        if lines_v:
            bus_v = lines_v[0]
        else:
            bus_v = 0.0

        gen_hot = indices_to_hot(grid.bus.gen[bus_id], length=n_gen, dtype=np.float)
        load_hot = indices_to_hot(grid.bus.load[bus_id], length=n_load, dtype=np.float)
        gen_inj = np.concatenate((obs.prod_p * gen_hot, obs.prod_q * gen_hot))
        load_inj = np.concatenate((obs.load_p * load_hot, obs.load_q * load_hot))

        node_features[bus_id, :] = np.concatenate(([bus_v], gen_inj, load_inj))

    global_features = global_features.astype(np.float)
    node_features = node_features.astype(np.float)
    edge_features = edge_features.astype(np.float)

    graph_dict = {
        "globals": global_features,
        "nodes": node_features,
        "edges": edge_features,
        "senders": senders,
        "receivers": receivers,
    }

    return graph_dict


def obs_to_graph_dict_by_grid(obs, grid):
    """
    Convert observation to a graph network data dictionary.

    Data dictionary consists of:
        globals: np.ndarray (n_globals, )
            Represents global attributes of a graph.

        nodes: np.ndarray (n_nodes, n_node_features)
            Represents feature vector for each node in the graph.

        edges: np.ndarray (n_edges, n_edge_features)
            Represents feature vector for each edge in the graph.


    Observation attributes are:
        year, month, day, hour_of_day, minute_of_hour, day_of_week:
            Included in globals. year and minute_of_hour omitted.

        prod_p, prod_q, prod_v:
            Included in node features.

        load_p, load_q, load_v:
            Included in node features.

        p_or, q_or, v_or, a_or, p_ex, q_ex, v_ex, a_ex:
            Included in edge features.

        rho:
            Included in edge features.

        topo_vect:
            Included in edge connections.

        line_status:
            Included as a mask on edge values.

        time_before_cooldown_sub:
            Not included. Can be added to node features.

        time_before_cooldown_line, timestep_overflow, time_next_maintenance, duration_next_maintenance:
            Not included. Can be added to edge features.

        target_dispatch, actual_dispatch:
            Not included.

    """

    n_sub = grid.n_sub
    n_line = grid.n_line
    n_gen = grid.n_gen
    n_load = grid.n_load

    # Features
    global_features = np.zeros((4,))
    if not is_nonetype(obs):
        global_features = np.array(
            [obs.month, obs.day, obs.day_of_week, obs.hour_of_day]
        )

    edges_or = np.zeros((n_line, 13))
    if not is_nonetype(obs):
        edges_or[:, 0] = obs.p_or
        edges_or[:, 1] = obs.q_or
        edges_or[:, 2] = obs.a_or
        edges_or[:, 3] = obs.v_or
        edges_or[:, 4] = obs.p_ex
        edges_or[:, 5] = obs.q_ex
        edges_or[:, 6] = obs.a_ex
        edges_or[:, 7] = obs.v_ex
        edges_or[:, 8] = obs.rho

        # Topology
        edges_or[:, 9] = np.equal(grid.line.sub_bus_or, 1).astype(np.float)
        edges_or[:, 10] = np.equal(grid.line.sub_bus_or, 2).astype(np.float)
        edges_or[:, 11] = np.equal(grid.line.sub_bus_ex, 1).astype(np.float)
        edges_or[:, 12] = np.equal(grid.line.sub_bus_ex, 2).astype(np.float)
        edges_or[:, 9:] = np.multiply(
            edges_or[:, 9:], np.atleast_2d(obs.line_status.astype(np.float)).T
        )

    edges_ex = edges_or
    edge_features = np.concatenate((edges_or, edges_ex), axis=0)

    node_features = np.zeros((n_sub, 2 * 2 * (n_gen + n_load)))
    if not is_nonetype(obs):
        for sub_id in grid.sub.index:

            # Generator features
            sub_mask = np.equal(grid.gen["sub"], sub_id).astype(np.float)

            sub_bus_mask = sub_mask * np.equal(grid.gen["sub_bus"], 1).astype(np.float)
            gen_inj_sub_bus_1 = np.concatenate(
                (obs.prod_p * sub_bus_mask, obs.prod_q * sub_bus_mask)
            )

            sub_bus_mask = sub_mask * np.equal(grid.gen["sub_bus"], 2).astype(np.float)
            gen_inj_sub_bus_2 = np.concatenate(
                (obs.prod_p * sub_bus_mask, obs.prod_q * sub_bus_mask)
            )

            # Load features
            sub_mask = np.equal(grid.load["sub"], sub_id).astype(np.float)

            sub_bus_mask = sub_mask * np.equal(grid.load["sub_bus"], 1).astype(np.float)
            load_inj_sub_bus_1 = np.concatenate(
                (obs.load_p * sub_bus_mask, obs.load_q * sub_bus_mask)
            )

            sub_bus_mask = sub_mask * np.equal(grid.load["sub_bus"], 2).astype(np.float)
            load_inj_sub_bus_2 = np.concatenate(
                (obs.load_p * sub_bus_mask, obs.load_q * sub_bus_mask)
            )

            gen_inj = np.concatenate((gen_inj_sub_bus_1, gen_inj_sub_bus_2))
            load_inj = np.concatenate((load_inj_sub_bus_1, load_inj_sub_bus_2))
            node_features[sub_id, :] = np.concatenate((gen_inj, load_inj))

    # Double edges
    senders_or = grid.line.sub_or.values
    receivers_or = grid.line.sub_ex.values
    senders_ex = grid.line.sub_ex.values
    receivers_ex = grid.line.sub_or.values

    # Edges from substation to substation
    senders = np.concatenate((senders_or, senders_ex), axis=0)
    receivers = np.concatenate((receivers_or, receivers_ex), axis=0)

    # Data-types
    global_features = global_features.astype(np.float)
    node_features = node_features.astype(np.float)
    edge_features = edge_features.astype(np.float)
    senders = senders.astype(np.int)
    receivers = receivers.astype(np.int)

    graph_dict = {
        "globals": global_features,
        "nodes": node_features,
        "edges": edge_features,
        "senders": senders,
        "receivers": receivers,
    }

    return graph_dict


def obses_to_lgraphs(obses, dones, case, n_window=1, max_length=-1):
    grid = GridDCOPF(case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p)

    single_graphs_dict_list = []
    grid.update(obses[0], reset=True)
    for i, (obs, done) in enumerate(zip(obses, dones)):
        grid.update(obs, reset=done)

        graph_dict = obs_to_graph_dict_by_grid(obs, grid)
        single_graphs_dict_list.append(graph_dict)

        if i % 1000 == 0:
            print(f"{i}/{len(obses)}")

        if 0 < max_length == (i + 1):
            break

    graph_dict_padding = [obs_to_graph_dict_by_grid(None, grid)] * n_window
    graph_dict_queue = deque(graph_dict_padding)

    graphs_dict_list = []
    for single_graph_dict in single_graphs_dict_list:
        graph_dict_queue.popleft()
        graph_dict_queue.append(single_graph_dict)

        graph_dict = dict()
        graph_dict["senders"] = graph_dict_queue[-1]["senders"]
        graph_dict["receivers"] = graph_dict_queue[-1]["receivers"]

        graph_dict["globals"] = np.hstack([g_d["globals"] for g_d in graph_dict_queue])
        graph_dict["nodes"] = np.hstack([g_d["nodes"] for g_d in graph_dict_queue])
        graph_dict["edges"] = np.hstack([g_d["edges"] for g_d in graph_dict_queue])

        graphs_dict_list.append(graph_dict)

    return graphs_dict_list


def obses_to_cgraphs(obses, dones, case, n_window=1, max_length=-1):
    graphs_dict_list = obses_to_lgraphs(
        obses, dones, case, n_window=n_window, max_length=max_length
    )
    cgraphs = lgraphs_to_cgraphs(graphs_dict_list)

    return cgraphs


def combined_dict_to_dict_list(combined_dict):
    n_elements = None

    fields = combined_dict.keys()
    for field in fields:
        field_values = combined_dict[field]
        if field_values:
            n_elements = len(field_values)
            break

    dict_list = []
    for i in range(n_elements):
        graph_dict = dict()
        for field in fields:
            graph_dict[field] = combined_dict[field][i]

        dict_list.append(graph_dict)

    return dict_list


def lgraphs_to_cgraphs(dict_list):
    combined_dict = {
        "globals": [],
        "edges": [],
        "nodes": [],
        "senders": [],
        "receivers": [],
    }
    for graph_dict in dict_list:
        for field in graph_dict:
            combined_dict[field].append(graph_dict[field])

    for field in combined_dict:
        if not len(combined_dict[field]):
            combined_dict[field] = None

    return combined_dict
