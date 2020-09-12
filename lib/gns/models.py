import graph_nets as gn
import sonnet as snt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from lib.data_utils import is_nonetype


class GraphNetworkSwitching(snt.Module):
    def __init__(
        self,
        n_global_features,
        n_node_features,
        n_edge_features,
        n_hidden=(16, 16),
        n_message_passes=3,
        pos_class_weight=1.0,
        learning_rate=None,
        graphs_signature=None,
        labels_signature=None,
    ):
        super(GraphNetworkSwitching, self).__init__()

        # Output graph feature dimensions
        self.n_global_features = n_global_features
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        # Graph network
        self.n_message_passes = n_message_passes
        self.n_hidden = n_hidden
        self.graph_network = self._build_graph_network()

        self.pos_class_weight = pos_class_weight

        self.learning_rate = learning_rate
        self.optimizer = snt.optimizers.Adam(self.learning_rate)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(
            name="weighted_binary_crossentropy"
        )

        # Compiling tf function
        if not is_nonetype(graphs_signature) and not is_nonetype(labels_signature):
            self.tf_compile_functions(graphs_signature, labels_signature)

    def _build_graph_network(self):
        return gn.modules.GraphNetwork(
            global_model_fn=self._build_global_model_fn(),
            node_model_fn=self._build_node_model_fn(),
            edge_model_fn=self._build_edge_model_fn(),
            reducer=tf.math.unsorted_segment_mean,
            name="graph_network",
        )

    def _build_global_model_fn(self):
        return lambda: snt.nets.MLP(
            output_sizes=[*self.n_hidden, self.n_global_features],
            activate_final=False,
            name="global_model",
        )

    def _build_node_model_fn(self):
        return lambda: snt.nets.MLP(
            output_sizes=[*self.n_hidden, self.n_node_features], name="node_model"
        )

    def _build_edge_model_fn(self):
        return lambda: snt.nets.MLP(
            output_sizes=[*self.n_hidden, self.n_edge_features], name="edge_model"
        )

    def graph_forward_pass(self, input_graphs):
        for _ in range(self.n_passes):
            input_graphs = self.graph_network(input_graphs)
        return input_graphs

    def __call__(self, input_graphs):
        output_graphs = self.graph_forward_pass(input_graphs)
        return output_graphs

    def train_step(self, input_graphs, labels):
        with tf.GradientTape() as gt:
            output_graphs = self.__call__(input_graphs)
            logits = output_graphs.globals[:, 0]
            probabilities = tf.math.sigmoid(logits, name="probabilities")

            loss = self.loss_fn(labels, probabilities)

        predicted_labels = tf.math.greater_equal(probabilities, 0.5)
        predicted_labels = tf.cast(
            predicted_labels, dtype=tf.int32, name="predicted_labels"
        )

        # Update weights
        gradients = gt.gradient(loss, self.trainable_variables)
        self.optimizer.apply(gradients, self.trainable_variables)

        return output_graphs, loss, probabilities, predicted_labels, gradients

    def tf_compile_functions(self, graphs_signature, labels_signature):
        self.graph_forward_pass = tf.function(
            self.graph_forward_pass, input_signature=[graphs_signature]
        )
        self.__call__ = tf.function(self.__call__, input_signature=[graphs_signature])

        self.train_step = tf.function(
            self.train_step, input_signature=[graphs_signature, labels_signature]
        )


class GraphNetworkKeras(keras.Model):
    def __init__(
        self,
        n_globals,
        n_nodes,
        n_edges,
        n_passes,
        learning_rate,
        pos_class_weight,
        graphs_signature=None,
        labels_signature=None,
    ):
        super(GraphNetworkKeras, self).__init__()
        self.n_globals = n_globals
        self.n_nodes = n_nodes
        self.n_edges = n_edges

        self.global_model = None
        self.node_model = None
        self.edge_model = None

        self.n_passes = n_passes
        self.graph_network = self._build_graph_network()

        self.pos_class_weight = pos_class_weight

        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(self.learning_rate)

        self.loss_fn = keras.losses.BinaryCrossentropy(
            name="weighted_binary_crossentropy"
        )

        # Compiling tf function
        if not is_nonetype(graphs_signature) and not is_nonetype(labels_signature):
            self.tf_compile_functions(graphs_signature, labels_signature)

    def _build_graph_network(self):
        return gn.modules.GraphNetwork(
            global_model_fn=self._build_global_model_fn(),
            node_model_fn=self._build_node_model_fn(),
            edge_model_fn=self._build_edge_model_fn(),
            reducer=tf.math.unsorted_segment_mean,
        )

    def _build_global_model_fn(self):
        self.global_model = tf.keras.Sequential(
            [
                layers.Dense(
                    16,
                    input_shape=(self.n_edges + self.n_nodes + self.n_globals,),
                    activation="relu",
                ),
                layers.Dense(16, activation="relu"),
                layers.Dense(self.n_globals, activation=None),
            ]
        )
        return lambda: self.global_model

    def _build_node_model_fn(self):
        self.node_model = keras.Sequential(
            [
                layers.Dense(
                    16,
                    input_shape=(self.n_edges + self.n_nodes + self.n_globals,),
                    activation="relu",
                ),
                layers.Dense(16, activation="relu"),
                layers.Dense(self.n_nodes, activation=None),
            ]
        )
        return lambda: self.node_model

    def _build_edge_model_fn(self):
        self.edge_model = keras.Sequential(
            [
                layers.Dense(
                    16,
                    input_shape=(self.n_edges + 2 * self.n_nodes + self.n_globals,),
                    activation="relu",
                ),
                layers.Dense(16, activation="relu"),
                layers.Dense(self.n_edges, activation=None),
            ]
        )
        return lambda: self.edge_model

    def graph_forward_pass(self, input_graphs):
        for _ in range(self.n_passes):
            input_graphs = self.graph_network(input_graphs)
        return input_graphs

    def call(self, inputs, training=None, mask=None):
        input_graphs = inputs
        output_graphs = self.graph_forward_pass(input_graphs)
        return output_graphs

    def train_step(self, data):
        input_graphs, labels = data
        with tf.GradientTape() as gt:
            output_graphs = self.__call__(input_graphs)
            logits = output_graphs.globals[:, 0]
            probabilities = tf.math.sigmoid(logits, "probabilities")

            loss = self.loss_fn(labels, probabilities)

        predicted_labels = tf.math.greater_equal(probabilities, 0.5)
        predicted_labels = tf.cast(
            predicted_labels, dtype=tf.int32, name="predicted_labels"
        )

        # Update weights
        gradients = gt.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return output_graphs, loss, probabilities, predicted_labels, gradients

    def tf_compile_functions(self, graphs_signature, labels_signature):
        self.graph_forward_pass = tf.function(
            self.graph_forward_pass, input_signature=[graphs_signature]
        )
        self.__call__ = tf.function(self.__call__, input_signature=[graphs_signature])

        self.train_step = tf.function(
            self.train_step, input_signature=[graphs_signature, labels_signature]
        )
