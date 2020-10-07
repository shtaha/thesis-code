import numpy as np

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
        name = "/".join(var.name.split("/")[-3:])

        n_params = np.prod(var.shape)
        pprint(
            name,
            "{:>10}".format(str(var.shape)),
            "{:>10}".format(n_params),
            "{:>10.2f}".format(np.linalg.norm(var.numpy())),
        )
        n_params_total = n_params_total + n_params

    print("-" * (40 + 5 * 10))
    pprint("Total params:", " " * 10, "{:>10}".format(n_params_total))


def print_gradient_norm(grads, variables):
    for var, grad in zip(variables, grads):
        name = "/".join(var.name.split("/")[-3:])
        pprint(f"    - {name}", str(grad.shape), np.linalg.norm(grad.numpy()), shift=60)
