import numpy as np
import tensorflow as tf

from ..visualizer import pprint


def print_variables(variables):
    n_params_total = 0
    pprint(
        "Name",
        "{:>10}".format("Shape"),
        "{:>10}".format("Param #"),
        "{:>10}".format("L2-Norm"),
    )
    for var in variables:
        n_params = np.prod(var.shape)
        pprint(
            var.name,
            "{:>10}".format(str(var.shape)),
            "{:>10}".format(n_params),
            "{:>10.2f}".format(np.linalg.norm(var.numpy())),
        )
        n_params_total = n_params_total + n_params

    print("-" * (40 + 5 * 10))
    pprint("Total params:", " " * 10, "{:>10}".format(n_params_total))


def gradient_norm(grads, var_names=None):
    for i, grad in enumerate(grads):
        if var_names:
            var_name = var_names[i]
        else:
            var_name = f"var_{i}"

        grad_norm = tf.linalg.norm(grad)

        pprint(var_name, "{:>10}".format(grad_norm.numpy()))
