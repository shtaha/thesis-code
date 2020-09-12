import bz2
import datetime
import os
from io import StringIO

import numpy as np
import pandas as pd


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

    if dtype == np.bool:
        hot = np.True_
    else:
        hot = 1

    if len(hot_indices):
        vector[hot_indices] = hot
    return vector


def hot_to_indices(bool_array: np.ndarray) -> np.ndarray:
    """
    Only works for 1D-vector.
    """
    index_array = np.flatnonzero(bool_array)
    return index_array


def read_bz2_to_dataframe(file_path, sep=";"):
    data_csv = bz2.BZ2File(file_path).read().decode()
    return pd.read_csv(StringIO(data_csv), sep=sep)


def env_pf(env_dc):
    return "dc" if env_dc else "ac"


def is_nonetype(obj):
    return isinstance(obj, type(None))
