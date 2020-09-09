import graph_nets as gn
import tensorflow as tf
from graph_nets import utils_tf


def equal_graphs(graphs_a, graphs_b):
    cond = tf.constant([True])

    v_a = utils_tf.get_num_graphs(graphs_a)
    v_b = utils_tf.get_num_graphs(graphs_b)
    cond = tf.math.logical_and(cond, v_a == v_b)

    for field in [
        "n_node",
        "n_edge",
        "globals",
        "nodes",
        "edges",
        "receivers",
        "senders",
    ]:
        v_a = getattr(graphs_a, field)
        v_b = getattr(graphs_b, field)

        check = tf.reduce_all(tf.equal(v_a, v_b))
        cond = tf.math.logical_and(cond, check)

        check = tf.reduce_all(tf.equal(v_a.shape, v_b.shape))
        cond = tf.math.logical_and(cond, check)

        if not cond:
            print(field)
            print(v_a)
            print(v_b)

    if utils_tf.get_num_graphs(graphs_a) > 1:
        for graph_idx in range(utils_tf.get_num_graphs(graphs_a)):
            graph_a = utils_tf.get_graph(graphs_a, graph_idx)
            graph_b = utils_tf.get_graph(graphs_b, graph_idx)
            check = equal_graphs(graph_a, graph_b)
            cond = tf.math.logical_and(cond, check)
            if not cond:
                print(graph_a)
                print(graph_b)

    return cond


def stack_graph_field(field_unstacked):
    field_stacked = None
    if not isinstance(field_unstacked, type(None)):
        field_stacked = tf.concat(tf.unstack(field_unstacked), axis=0)
    return field_stacked


def reenumerate_graph_edges(node_ids, n_edge, n_node):
    if len(n_edge.shape) > 0:
        result = node_ids + tf.repeat(
            tf.multiply(tf.range(n_edge.shape[0]), n_node), repeats=n_edge
        )
    else:
        result = node_ids
    return result


def graph_dict_to_graph(graph_dict, label):
    return utils_tf.data_dicts_to_graphs_tuple([graph_dict]), label


def stack_graphs(graphs):
    graphs = graphs.map(stack_graph_field, fields=("globals", "nodes", "edges"))
    graphs = graphs.map(stack_graph_field, fields=("receivers", "senders"))

    if utils_tf.get_num_graphs(graphs) > 1:
        graphs = graphs.map(lambda x: tf.squeeze(x), fields=("n_node", "n_edge"))
    else:
        graphs = graphs.map(lambda x: tf.reshape(x, (1,)), fields=("n_node", "n_edge"))

    graphs_stacked = gn.graphs.GraphsTuple(
        nodes=graphs.nodes,
        edges=graphs.edges,
        globals=graphs.globals,
        receivers=reenumerate_graph_edges(
            graphs.receivers, graphs.n_edge, graphs.n_node
        ),
        senders=reenumerate_graph_edges(graphs.senders, graphs.n_edge, graphs.n_node),
        n_node=graphs.n_node,
        n_edge=graphs.n_edge,
    )

    return graphs_stacked
