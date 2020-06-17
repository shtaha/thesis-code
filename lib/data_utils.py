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

        bus_names = []
        for bus_id, bus_name in zip(grid.bus.index, grid.bus["name"]):
            sub_id = bus_name.split("_")[-1]
            bus_names.append(f"bus-{bus_id}-{sub_id}")

        grid.bus["name"] = bus_names
        grid.line["name"] = env.name_line
        grid.gen["name"] = env.name_gen
        grid.load["name"] = env.name_load

    elif env.name == "l2rpn_2019":
        bus_names = [
            f"bus-{bus_id}-{sub_id}"
            for bus_id, sub_id in zip(grid.bus.index, grid.bus["name"])
        ]

        grid.bus["name"] = bus_names
        grid.line["name"] = env.name_line
        grid.gen["name"] = env.name_gen
        grid.load["name"] = env.name_load

    # Environment and backend inconsistency
    grid.gen["min_p_mw"] = env.gen_pmin
    grid.gen["max_p_mw"] = env.gen_pmax
    grid.gen["type"] = env.gen_type
    grid.gen["gen_redispatchable"] = env.gen_redispatchable
    grid.gen["gen_max_ramp_up"] = env.gen_max_ramp_up
    grid.gen["gen_max_ramp_down"] = env.gen_max_ramp_down
    grid.gen["gen_min_uptime"] = env.gen_min_uptime
    grid.gen["gen_min_downtime"] = env.gen_min_downtime

    if verbose:
        print(env.name.upper())
        print("bus\n" + grid.bus.to_string())
        print("gen\n" + grid.gen.to_string())
        print("load\n" + grid.load.to_string())
        print("line\n" + grid.line.to_string())


def parse_gurobi_log(log):
    gap = 0.0
    for line in log.split("\n")[-5:]:
        if "Best objective" in line:
            gap = float(line.strip().split()[-1].replace("%", ""))

    return {"gap": gap}
