import bz2
import datetime
import os
from collections import deque
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


def extract_target_windows(targets, mask=None, n_window=0):
    window = np.zeros_like(targets)
    for i in range(len(targets)):
        start = np.maximum(i - n_window, 0)
        end = i + n_window + 1
        window[i] = targets[start:end].any()

    window = window.astype(np.bool)
    if not is_nonetype(mask):
        window = np.logical_and(window, mask)

    return window


def extract_history_windows(targets, n_window=0):
    window = np.zeros_like(targets)
    for i in range(len(targets)):
        start = i
        end = i + n_window + 1

        window[i] = targets[start:end].any()

    return window.astype(np.bool)


def moving_window(
    items, mask_targets=None, n_window=0, process_fn=None, combine_fn=None, padding=None
):

    if is_nonetype(mask_targets):
        mask_targets = np.ones_like(items, dtype=np.bool)

    mask_history = extract_history_windows(mask_targets, n_window=n_window)

    if is_nonetype(process_fn):

        def process_fn(x):
            return x

    if is_nonetype(combine_fn):

        def combine_fn(x):
            return x[-1]

    padding_window = [padding] * (n_window + 1)
    queue = deque(padding_window)

    history = []
    for i, item in enumerate(items):
        if mask_history[i]:
            queue.popleft()

            pitem = process_fn(item)
            queue.append(pitem)

            if mask_targets[i]:
                assert len(queue) == n_window + 1
                history.append(combine_fn(list(queue)))

    assert len(history) == mask_targets.sum()
    return history
