import graph_nets as gn
import sonnet as snt
import tensorflow as tf


class GraphNetwork(snt.Module):
    def __init__(
        self,
        n_global_features,
        n_node_features,
        n_edge_features,
        n_edges,
        n_nodes,
        n_hidden_global,
        n_hidden_node,
        n_hidden_edge,
        dropout_rate=0.1,
        n_message_passes=3,
    ):
        super(GraphNetwork, self).__init__()

        # Output graph feature dimensions
        self.n_global_features = n_global_features
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_nodes = n_nodes
        self.n_edges = n_edges

        # Graph network
        self.n_message_passes = n_message_passes
        self.n_hidden_global = n_hidden_global
        self.n_hidden_node = n_hidden_node
        self.n_hidden_edge = n_hidden_edge

        self.dropout_rate = dropout_rate
        self.graph_network = self._build_graph_network()

    def _build_graph_network(self):
        return gn.modules.GraphNetwork(
            global_model_fn=self._build_global_model_fn(),
            node_model_fn=self._build_node_model_fn(),
            edge_model_fn=self._build_edge_model_fn(),
            reducer=tf.math.unsorted_segment_mean,
        )

    def _build_global_model_fn(self):
        return lambda: snt.nets.MLP(
            output_sizes=[*self.n_hidden_global, self.n_global_features],
            activation=tf.nn.relu,
            # dropout_rate=self.dropout_rate,
            name="global_model",
        )

    def _build_node_model_fn(self):
        return lambda: snt.nets.MLP(
            output_sizes=[*self.n_hidden_node, self.n_node_features],
            activation=tf.nn.relu,
            # dropout_rate=self.dropout_rate,
            name="node_model",
        )

    def _build_edge_model_fn(self):
        return lambda: snt.nets.MLP(
            output_sizes=[*self.n_hidden_edge, self.n_edge_features],
            activation=tf.nn.relu,
            # dropout_rate=self.dropout_rate,
            name="edge_model",
        )

    def __call__(self, input_graphs, is_training=None):
        for _ in range(self.n_message_passes):
            input_graphs = self.graph_network(input_graphs)
        return input_graphs
