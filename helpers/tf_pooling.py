import tensorflow as tf
import numpy as np

n_batch = 2
n_edges = 2
n_features = 5

a = np.random.randint(0, 10, (n_batch, n_edges, n_features))
a = tf.constant(a)

b = tf.nn.max_pool(
    a, ksize=1, strides=1, padding="SAME"
)

c = tf.math.reduce_max(
    a, axis=1
)


print(a.shape)
print(b.shape)
print(c.shape)
