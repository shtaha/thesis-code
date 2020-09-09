import numpy as np


def print_variables(variables):
    for var in variables:
        print(var.name, var.shape, np.linalg.norm(var.numpy()))
