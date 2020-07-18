import datetime
import os

import numpy as np


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


def create_results_dir(results_dir_path, model_name=None):
    if not model_name:
        model_name = "results"

    results_dir = make_dir(
        os.path.join(
            results_dir_path,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_" + model_name),
        )
    )
    return results_dir


def indices_to_hot(
    hot_indices: np.ndarray, length: np.int, dtype=np.bool
) -> np.ndarray:
    """
    Only works for 1D-vector.
    """
    vector = np.zeros((length,), dtype=dtype)
    vector[hot_indices] = np.True_
    return vector


def hot_to_indices(bool_array: np.ndarray) -> np.ndarray:
    """
    Only works for 1D-vector.
    """
    index_array = np.flatnonzero(bool_array)
    return index_array


def parse_gurobi_log(log):
    gap = 0.0
    for line in log.split("\n")[-5:]:
        if "Best objective" in line:
            gap = float(line.strip().split()[-1].replace("%", ""))

    return {"gap": gap}
