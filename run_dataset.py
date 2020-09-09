import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 100

path = tf.keras.utils.get_file("mnist.npz", DATA_URL)
with np.load(path) as data:
    train_examples = data["x_train"]
    train_labels = data["y_train"]
    test_examples = data["x_test"]
    test_labels = data["y_test"]

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)
train_dataset = train_dataset.repeat(1)
test_dataset = test_dataset.repeat(1)

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Iterate over batches
fig, ax = plt.subplots()

for x_batch, y_batch in train_dataset:
    print(type(x_batch), x_batch.shape)
    print(type(y_batch), y_batch.shape)
    for x, y in zip(x_batch, y_batch):
        print(type(x), x.shape)
        print(type(y), y.shape)
        ax.imshow(x)
        ax.set_title(f"Label: {y}")
        fig.show()
    break

train_iterator = iter(train_dataset)
test_iterator = iter(test_dataset)

x_batch, y_batch = train_iterator.get_next()
print(type(x_batch), x_batch.shape)
print(type(y_batch), y_batch.shape)
