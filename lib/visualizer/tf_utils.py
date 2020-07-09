import numpy as np


def print_trainable_variables(model):
    for var in model.trainable_variables:
        print(var.name, var.shape, np.linalg.norm(var.numpy()))
