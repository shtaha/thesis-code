import tensorflow as tf


class ResidulaFCBlock(tf.keras.layers.Layer):
    def __init__(self, n_hidden, activation="relu", name=None, **kwargs):
        super(ResidulaFCBlock, self).__init__(name=name)

        self.dense_1 = tf.keras.layers.Dense(n_hidden, activation=None, **kwargs)
        self.dense_2 = tf.keras.layers.Dense(n_hidden, activation=None, **kwargs)

        self.activation = tf.keras.layers.Activation(activation)

    def call(self, input_tensor, training=False):
        x = self.dense_1(input_tensor)
        x = self.activation(x)

        x = self.dense_2(x)
        x = x + input_tensor
        x = self.activation(x)
        return x
