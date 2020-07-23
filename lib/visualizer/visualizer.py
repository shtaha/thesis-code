import json

import numpy as np


def print_matrix(matrix, name=None, spacing=None, decimals=4):
    if name:
        if type(matrix) == np.ndarray:
            shape = matrix.shape
        else:
            shape = len(matrix)

        print(name, "=", str(shape), str(type(matrix)))

    lines = []
    matrix = np.squeeze(matrix)
    matrix = np.atleast_2d(matrix)

    max_value = np.max(np.abs(matrix))
    if not spacing:
        if max_value > 0 and not np.isinf(max_value):
            spacing = max([int(np.log10(max_value)) + 5, 6])
        else:
            spacing = 6

    for row in matrix:
        line = ""
        for cell in row:
            if not np.isinf(np.abs(cell)):
                if cell == 0 or np.abs(int(cell) - cell) < 1e-12:
                    pattern = "{:>" + str(int(spacing)) + "}"
                    line = line + pattern.format(int(cell))
                else:
                    pattern = "{:>" + str(int(spacing)) + "." + str(int(decimals)) + "}"
                    line = line + pattern.format(cell)
            else:
                pattern = "{:>" + str(int(spacing)) + "}"
                line = line + pattern.format(cell)

        lines.append(line)
    print("\n".join(lines))
    print("\n")


def print_dict(dictionary):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    print(json.dumps(dictionary, indent=1, cls=NumpyEncoder))


def pprint(*args):
    if len(args) < 2:
        raise ValueError("At least two arguments for printing.")

    format_str = "{:<35}" + "\t".join(["{}" for _ in range(len(args) - 1)])
    print(format_str.format(*args))
