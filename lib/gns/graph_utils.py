import graph_nets as gn
import numpy as np
import tensorflow as tf
from graph_nets import utils_tf

from ..visualizer import pprint


def equal_graphs(graphs_a, graphs_b, verbose=True):
    cond = tf.constant([True])

    v_a = utils_tf.get_num_graphs(graphs_a)
    v_b = utils_tf.get_num_graphs(graphs_b)
    cond = tf.math.logical_and(cond, v_a == v_b)

    if not cond and verbose:
        pprint("num_graphs", cond.numpy())

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

        check_values = tf.reduce_all(tf.equal(v_a, v_b))
        cond = tf.math.logical_and(cond, check_values)

        check_shape = tf.reduce_all(tf.equal(v_a.shape, v_b.shape))
        cond = tf.math.logical_and(cond, check_shape)

        if not cond and verbose:
            pprint(
                field,
                f"values = {check_values.numpy()}",
                f"shape = {check_shape.numpy()}",
                cond.numpy(),
            )

    if verbose:
        print()

    if utils_tf.get_num_graphs(graphs_a) > 1:
        for graph_idx in range(utils_tf.get_num_graphs(graphs_a)):
            graph_a = utils_tf.get_graph(graphs_a, graph_idx)
            graph_b = utils_tf.get_graph(graphs_b, graph_idx)
            check = equal_graphs(graph_a, graph_b, verbose=False)
            cond = tf.math.logical_and(cond, check)

    return cond


def check_lgraphs(lgraphs):
    fields = set()
    dims = dict()
    for dgraph in lgraphs:
        f = sorted(list(dgraph.keys()))
        fields.update(f)

    for field in fields:
        dims[field] = []

    for dgraph in lgraphs:
        f = sorted(list(dgraph.keys()))
        assert len(f) == len(fields)
        for field in fields:
            assert field in f
            d = dgraph[field].shape
            dims[field].append(str(d))

    for field in fields:
        assert len(np.unique(dims[field])) == 1


def dgraph_to_graph(dgraph):
    return utils_tf.data_dicts_to_graphs_tuple([dgraph])


def dgraphs_to_graphs(dgraphs):
    return utils_tf.data_dicts_to_graphs_tuple(dgraphs)


def stack_batch(graph_batch, n_global_features, n_node_features, n_edge_features):
    globals_features = tf.reshape(graph_batch.globals, shape=[-1, n_global_features])
    nodes_features = tf.reshape(graph_batch.nodes, shape=[-1, n_node_features])
    edges_features = tf.reshape(graph_batch.edges, shape=[-1, n_edge_features])

    receivers = tf.reshape(graph_batch.receivers, shape=[-1])
    senders = tf.reshape(graph_batch.senders, shape=[-1])

    n_node = tf.reshape(graph_batch.n_node, shape=[-1])
    n_edge = tf.reshape(graph_batch.n_edge, shape=[-1])

    mask = tf.repeat(
        tf.multiply(tf.range(utils_tf.get_num_graphs(graph_batch)), n_node),
        repeats=n_edge,
    )
    receivers = receivers + mask
    senders = senders + mask

    graph_batch = gn.graphs.GraphsTuple(
        nodes=nodes_features,
        edges=edges_features,
        globals=globals_features,
        receivers=receivers,
        senders=senders,
        n_node=n_node,
        n_edge=n_edge,
    )

    return graph_batch


def tf_graph_dataset(cgraphs):
    graph_dataset = tf.data.Dataset.from_tensor_slices(cgraphs)
    graph_dataset = graph_dataset.map(dgraph_to_graph)
    return graph_dataset


def tf_batched_graph_dataset(
    cgraphs,
    n_global_features,
    n_node_features,
    n_edge_features,
    n_batch=1,
    stack_fn=stack_batch,
):
    graph_dataset = tf_graph_dataset(cgraphs)

    if n_batch == 1:
        return graph_dataset
    else:

        def stack_fn_params(graph_batch):
            return stack_fn(
                graph_batch, n_global_features, n_node_features, n_edge_features
            )

        graph_dataset = graph_dataset.batch(n_batch)
        return graph_dataset.map(stack_fn_params)


def get_graph_feature_dimensions(tgraphs=None, lgraphs=None, cgraphs=None):
    dimensions = dict()
    if tgraphs:
        dimensions = dict(
            n_global_features=tgraphs.globals.shape[-1],
            n_node_features=tgraphs.nodes.shape[-1],
            n_edge_features=tgraphs.edges.shape[-1],
        )

    if lgraphs:
        dimensions = dict(
            n_global_features=lgraphs[0]["globals"].shape[-1],
            n_node_features=lgraphs[0]["nodes"].shape[-1],
            n_edge_features=lgraphs[0]["edges"].shape[-1],
        )

    if cgraphs:
        dimensions = dict(
            n_global_features=cgraphs["globals"][0].shape[-1],
            n_node_features=cgraphs["nodes"][0].shape[-1],
            n_edge_features=cgraphs["edges"][0].shape[-1],
        )

    return dimensions
