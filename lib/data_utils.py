import datetime
import os

import numpy as np


def make_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

    return directory


def create_results_dir(results_dir_path):
    results_dir = make_dir(
        os.path.join(
            results_dir_path,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_model"),
        )
    )
    return results_dir


def hot_vector(hot_indices: np.ndarray, length: np.int, dtype=np.bool) -> np.ndarray:
    vector = np.zeros((length,), dtype=dtype)
    vector[hot_indices] = np.True_
    return vector
