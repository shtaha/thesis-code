import numpy as np
from graph_nets import graphs
from graph_nets import utils_tf

from ..data_utils import is_nonetype, moving_window


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


def obs_to_dgraph(obs, tc):
    if not is_nonetype(obs):
        lines_or_to_sub_bus = tc.lines_or_to_sub_bus(obs)
        lines_ex_to_sub_bus = tc.lines_ex_to_sub_bus(obs)

        # Edge features
        edges_or = []
        for sub_bus_ex in [1, 2]:
            for sub_bus_or in [1, 2]:
                mask = np.multiply(
                    tc.sub_bus_mask(lines_or_to_sub_bus, sub_bus_or, np.float),
                    tc.sub_bus_mask(lines_ex_to_sub_bus, sub_bus_ex, np.float),
                )
                p_or = np.multiply(obs.p_or, mask)
                edges_or.append(p_or)

        edges_or.append(obs.rho)

        # Add status, cooldown, maintenance, overflow
        # edges_or.append(obs.line_status.astype(np.float))
        # edges_or.append(obs.timestep_overflow)
        # edges_or.append(obs.time_next_maintenance)
        # edges_or.append(obs.duration_next_maintenance)
        # edges_or.append(obs.time_before_cooldown_line)

        edges_or = np.vstack(edges_or).T
        edges_ex = edges_or.copy()
        edge_features = np.concatenate((edges_or, edges_ex), axis=0)

        gens_to_sub_bus = tc.gens_to_sub_bus(obs)
        loads_to_sub_bus = tc.loads_to_sub_bus(obs)

        # Node features
        nodes = []
        prod_1 = np.multiply(tc.sub_bus_mask(gens_to_sub_bus, 1, np.float), obs.prod_p)
        prod_2 = np.multiply(tc.sub_bus_mask(gens_to_sub_bus, 2, np.float), obs.prod_p)
        load_1 = np.multiply(tc.sub_bus_mask(loads_to_sub_bus, 1, np.float), obs.load_p)
        load_2 = np.multiply(tc.sub_bus_mask(loads_to_sub_bus, 2, np.float), obs.load_p)

        for sub_id in range(tc.n_sub):
            p_1 = np.multiply(
                np.equal(tc.gen_to_sub_id, sub_id, dtype=np.float), prod_1
            )
            p_2 = np.multiply(
                np.equal(tc.gen_to_sub_id, sub_id, dtype=np.float), prod_2
            )

            l_1 = np.multiply(
                np.equal(tc.load_to_sub_id, sub_id, dtype=np.float), load_1
            )
            l_2 = np.multiply(
                np.equal(tc.load_to_sub_id, sub_id, dtype=np.float), load_2
            )

            nodes.append(np.concatenate((p_1, p_2, l_1, l_2)))

        # Add cooldown
        node_features = np.vstack(nodes)
        # node_features = np.append(
        #     node_features, np.atleast_2d(obs.time_before_cooldown_sub).T, axis=1
        # )
    else:
        # node_features = np.zeros((tc.n_sub, 2 * (tc.n_gen + tc.n_load) + 1))
        # edge_features = np.zeros((2 * tc.n_line, 4 + 6))

        node_features = np.zeros((tc.n_sub, 2 * (tc.n_gen + tc.n_load)))
        edge_features = np.zeros((2 * tc.n_line, 4 + 1))

    senders_or = tc.line_or_to_sub_id
    receivers_or = tc.line_ex_to_sub_id
    senders_ex = tc.line_ex_to_sub_id
    receivers_ex = tc.line_or_to_sub_id

    # Edges from substation to substation
    senders = np.concatenate((senders_or, senders_ex), axis=0).astype(np.int)
    receivers = np.concatenate((receivers_or, receivers_ex), axis=0).astype(np.int)

    edge_features = edge_features.astype(np.float)
    node_features = node_features.astype(np.float)

    from bclassification.utils_fc import obs_to_vect

    if not is_nonetype(obs):
        global_features = obs_to_vect(obs, tc)
    else:
        global_features = np.zeros(
            (2 * (tc.n_gen + tc.n_load) + tc.n_sub + 10 * tc.n_line,), dtype=np.float
        )

    global_features = global_features.astype(np.float)

    return {
        "globals": global_features,
        "nodes": node_features,
        "edges": edge_features,
        "senders": senders,
        "receivers": receivers,
    }


def obses_to_lgraphs(obses, tc, mask_targets=None, n_window=0):
    lgraphs = moving_window(
        obses,
        mask_targets=mask_targets,
        n_window=n_window,
        process_fn=obs_to_dgraph_with_tc(tc),
        combine_fn=dgraphs_to_dgraph,
        padding=obs_to_dgraph(None, tc),
    )

    return lgraphs


def obses_to_cgraphs(obses, tc, mask_targets=None, n_window=0):
    lgraphs = obses_to_lgraphs(obses, tc, mask_targets=mask_targets, n_window=n_window)
    cgraphs = lgraphs_to_cgraphs(lgraphs)
    return cgraphs


def cgraphs_to_lgraphs(combined_dict):
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


def dgraphs_to_dgraph(dgraphs):
    dgraph = dict()
    dgraph["senders"] = dgraphs[-1]["senders"]
    dgraph["receivers"] = dgraphs[-1]["receivers"]
    dgraph["globals"] = np.hstack([g_d["globals"] for g_d in dgraphs])
    dgraph["nodes"] = np.hstack([g_d["nodes"] for g_d in dgraphs])
    dgraph["edges"] = np.hstack([g_d["edges"] for g_d in dgraphs])
    return dgraph


def obs_to_dgraph_with_tc(tc):
    return lambda obs: obs_to_dgraph(obs, tc)
