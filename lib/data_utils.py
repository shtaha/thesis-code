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


def update_backend(env, verbose=False):
    """
    Update backend grid with missing data.
    """
    grid = env.backend._grid

    if env.name == "rte_case5_example":
        # Loads
        grid.load["controllable"] = False

        # Generators
        grid.gen["controllable"] = True
        grid.gen["min_p_mw"] = env.gen_pmin
        grid.gen["max_p_mw"] = env.gen_pmax

        # Additional data
        # Not used for the time being
        # grid.gen["type"] = env.gen_type
        # grid.gen["gen_redispatchable"] = env.gen_redispatchable
        # grid.gen["gen_max_ramp_up"] = env.gen_max_ramp_up
        # grid.gen["gen_max_ramp_down"] = env.gen_max_ramp_down
        # grid.gen["gen_min_uptime"] = env.gen_min_uptime
        # grid.gen["gen_min_downtime"] = env.gen_min_downtime

    if verbose:
        print(env.name.upper())
        print("bus\n" + grid.bus.to_string())
        print("gen\n" + grid.gen.to_string())
        print("load\n" + grid.load.to_string())
        print("line\n" + grid.line.to_string())
