import numpy as np
import tensorflow as tf


def print_variables(variables):
    for var in variables:
        print(var.name, var.shape, np.linalg.norm(var.numpy()))


def tf_train_test_split(dataset: tf.data.Dataset, dataset_size=None, test_frac=0.2):
    if not dataset_size:
        dataset_size = tf.data.experimental.cardinality(dataset).numpy()

    train_size = int((1 - test_frac) * dataset_size)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    return train_dataset, test_dataset


def tf_train_val_test_split(
    dataset: tf.data.Dataset, dataset_size=None, val_frac=0.15, test_frac=0.15
):
    assert val_frac + test_frac <= 1.0

    train_dataset, test_dataset = tf_train_test_split(
        dataset, dataset_size=dataset_size, test_frac=test_frac
    )
    train_dataset, val_dataset = tf_train_test_split(
        train_dataset, dataset_size=dataset_size, test_frac=val_frac / (1 - test_frac)
    )

    return train_dataset, val_dataset, test_dataset
